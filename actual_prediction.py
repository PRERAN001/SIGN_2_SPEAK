import torch
import json
import os
import cv2
import numpy as np
from app import WLASLVideoDataset
import logging
from tkinter import Tk, filedialog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_resnet3d(num_classes):
    import torchvision
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def preprocess_video(video_path, max_frames=16, img_size=224):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames found in the selected video.")
    # Uniformly sample max_frames from the video
    if len(frames) >= max_frames:
        idxs = np.linspace(0, len(frames)-1, max_frames).astype(int)
        frames = [frames[i] for i in idxs]
    else:
        while len(frames) < max_frames:
            frames.append(frames[-1])
    frames = np.stack(frames)  # [T, H, W, C]
    frames = frames.transpose(0, 3, 1, 2)  # [T, C, H, W]
    frames = torch.tensor(frames, dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames  # [T, C, H, W]

def main():
    # Load label encoder from dataset
    with open("wlasl_data/download_info.json", "r") as f:
        video_data = json.load(f)["data"]
    dataset = WLASLVideoDataset(video_data)
    num_classes = len(dataset.label_encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = get_resnet3d(num_classes).to(device)
    model.load_state_dict(torch.load("best_resnet3d_wlasl.pth", map_location=device))
    model.eval()

    # Open file dialog to select a video
    Tk().withdraw()  # Hide the main tkinter window
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.webm *.mkv *.flv *.wmv *.mpeg *.mpg")]
    )
    if not video_path:
        print("No video selected.")
        return
    print(f"Selected video: {video_path}")

    # Preprocess and predict
    frames = preprocess_video(video_path, max_frames=16, img_size=224)  # [T, C, H, W]
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    frames = frames.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    with torch.no_grad():
        outputs = model(frames)
        pred = torch.argmax(outputs, dim=1).item()
        pred_name = dataset.label_encoder.inverse_transform([pred])[0]
        print(f"Predicted label: {pred_name}")
        return pred_name

if __name__ == "__main__":
    main()