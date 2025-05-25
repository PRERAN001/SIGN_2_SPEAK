import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from app import WLASLVideoDataset
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_resnet3d(num_classes):
    import torchvision
    model = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    # Load existing video data
    with open("wlasl_data/download_info.json", "r") as f:
        video_data = json.load(f)["data"]

    dataset = WLASLVideoDataset(video_data)
    num_classes = len(dataset.label_encoder.classes_)

    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=dataset.encoded_labels, random_state=42
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = get_resnet3d(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(10):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            frames = batch['frames'].to(device)  # [B, T, C, H, W]
            labels = batch['labels'].to(device)
            frames = frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
        train_acc = 100. * train_correct / train_total
        logger.info(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                frames = batch['frames'].to(device)
                labels = batch['labels'].to(device)
                frames = frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
                outputs = model(frames)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_acc = 100. * val_correct / val_total
        logger.info(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet3d_wlasl.pth")
            logger.info("Saved new best model.")

if __name__ == "__main__":
    main()