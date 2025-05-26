import os
import json
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging
from pathlib import Path
import warnings
import urllib.request
import zipfile
from collections import defaultdict
import random
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from datetime import datetime
import dwani
import hashlib
from werkzeug.utils import secure_filename
import speech_recognition as sr
from text_sign import create_sign_video
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load configuration from environment variables
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
GEOAPIFY_API_KEY = os.getenv('GEOAPIFY_API_KEY')

# Configure Dwani API
dwani.api_key = os.getenv("DWANI_API_KEY")
dwani.api_base = os.getenv("DWANI_API_BASE_URL", "https://dwani-dwani-api.hf.space")

# Configure upload folders
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = BASE_DIR / 'Videos'
SIGN_VIDEOS_FOLDER = BASE_DIR / 'uploads'
AUDIO_FOLDER = BASE_DIR / 'static/audio'
SIGN_OUTPUT_FOLDER = BASE_DIR / 'sign_output'
COMBINED_OUTPUT_FOLDER = BASE_DIR / 'wlasl_data/combined_output'
TEMP_FOLDER = BASE_DIR / 'temp'

# Create necessary directories
for folder in [UPLOAD_FOLDER, SIGN_VIDEOS_FOLDER, AUDIO_FOLDER, SIGN_OUTPUT_FOLDER, COMBINED_OUTPUT_FOLDER, TEMP_FOLDER]:
    folder.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['SIGN_VIDEOS_FOLDER'] = str(SIGN_VIDEOS_FOLDER)
app.config['AUDIO_FOLDER'] = str(AUDIO_FOLDER)
app.config['SIGN_OUTPUT_FOLDER'] = str(SIGN_OUTPUT_FOLDER)
app.config['COMBINED_OUTPUT_FOLDER'] = str(COMBINED_OUTPUT_FOLDER)
app.config['TEMP_FOLDER'] = str(TEMP_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Cache for translations and speech
translation_cache = {}
speech_cache = {}

class WLASLDatasetDownloader:
    """Download and prepare WLASL dataset"""
    
    def __init__(self, data_dir="wlasl_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

    def download_metadata(self):
        """Download WLASL metadata"""
        metadata_path = self.data_dir / "WLASL_v0.3.json"
        if metadata_path.exists():
            logger.info("WLASL metadata already exists")
            return str(metadata_path)
        logger.info("Downloading WLASL metadata...")
        urllib.request.urlretrieve(self.metadata_url, metadata_path)
        logger.info(f"Metadata downloaded to {metadata_path}")
        return str(metadata_path)
    
    def load_metadata(self):
        """Load and parse WLASL metadata"""
        metadata_path = self.download_metadata()
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded metadata for {len(data)} words")
        return data
    
    def download_videos(self, max_words=50, max_videos_per_word=10):
        """Download a subset of WLASL videos"""
        metadata = self.load_metadata()
        videos_dir = self.data_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        downloaded_data = []
        word_count = 0
        
        for word_data in tqdm(metadata[:max_words], desc="Processing words"):
            if word_count >= max_words:
                break
            word = word_data['gloss']
            word_dir = videos_dir / word
            word_dir.mkdir(exist_ok=True)
            
            instances = word_data['instances']
            video_count = 0
            
            for instance in instances:
                if video_count >= max_videos_per_word:
                    break
                video_id = instance['video_id']
                url = instance['url']
                video_filename = f"{video_id}.mp4"
                video_path = word_dir / video_filename
                
                if video_path.exists():
                    downloaded_data.append({
                        'path': str(video_path),
                        'label': word,
                        'video_id': video_id
                    })
                    video_count += 1
                    continue
                
                try:
                    logger.info(f"Downloading {word} - {video_id}")
                    urllib.request.urlretrieve(url, video_path)
                    cap = cv2.VideoCapture(str(video_path))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if frame_count > 0:
                        downloaded_data.append({
                            'path': str(video_path),
                            'label': word,
                            'video_id': video_id
                        })
                        video_count += 1
                    else:
                        video_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to download {video_id}: {e}")
                    if video_path.exists():
                        video_path.unlink()
            
            if video_count > 0:
                word_count += 1
                logger.info(f"Downloaded {video_count} videos for '{word}'")
        
        # Filter classes with at least 2 samples
        label_counts = defaultdict(int)
        for item in downloaded_data:
            label_counts[item['label']] += 1
        
        min_samples = 2
        valid_labels = [label for label, count in label_counts.items() if count >= min_samples]
        filtered_data = [item for item in downloaded_data if item['label'] in valid_labels]
        
        logger.info(f"Filtered dataset to {len(filtered_data)} videos across {len(valid_labels)} classes (min {min_samples} samples per class)")
        
        # Save download info
        download_info = {
            'total_videos': len(filtered_data),
            'total_words': len(valid_labels),
            'data': filtered_data,
            'class_distribution': dict(label_counts)
        }
        info_path = self.data_dir / "download_info.json"
        with open(info_path, 'w') as f:
            json.dump(download_info, f, indent=2)
        
        logger.info(f"Final dataset: {len(filtered_data)} videos for {len(valid_labels)} words")
        return filtered_data

class WLASLVideoDataset(Dataset):
    """WLASL Video Dataset"""
    
    def __init__(self, video_data, max_frames=16, img_size=224):
        self.video_data = video_data
        self.max_frames = max_frames
        self.img_size = img_size
        self.video_paths = [item['path'] for item in video_data]
        self.labels = [item['label'] for item in video_data]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)
            cap.release()
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    black_frame = torch.zeros(3, self.img_size, self.img_size)
                    frames.append(black_frame)
            frames = frames[:self.max_frames]
            return torch.stack(frames)
        except Exception as e:
            logger.warning(f"Error processing video {video_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.encoded_labels[idx]
        frames = self.extract_frames(video_path)
        if frames is None:
            frames = torch.zeros(self.max_frames, 3, self.img_size, self.img_size)
        return {
            'frames': frames.float(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class Simple3DCNN(nn.Module):
    """Simple 3D CNN for video classification"""
    
    def __init__(self, num_classes, num_frames=16):
        super(Simple3DCNN, self).__init__()
        self.conv3d1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.conv3d3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.conv3d4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 1 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool3d(self.relu(self.conv3d1(x)))
        x = self.pool3d(self.relu(self.conv3d2(x)))
        x = self.pool3d(self.relu(self.conv3d3(x)))
        x = self.pool3d(self.relu(self.conv3d4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class WLASLTrainer:
    """WLASL Training class"""
    
    def __init__(self, data_dir="wlasl_data"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, max_words=50, max_videos_per_word=10):
        downloader = WLASLDatasetDownloader(self.data_dir)
        info_path = Path(self.data_dir) / "download_info.json"
        if info_path.exists():
            logger.info("Loading existing WLASL data...")
            with open(info_path, 'r') as f:
                download_info = json.load(f)
            video_data = download_info['data']
        else:
            logger.info("Downloading WLASL data...")
            video_data = downloader.download_videos(max_words, max_videos_per_word)
        logger.info(f"Prepared {len(video_data)} videos")
        return video_data
    
    def train(self, epochs=20, batch_size=4, learning_rate=1e-4, test_size=0.2, 
              max_words=50, max_videos_per_word=10):
        video_data = self.prepare_data(max_words, max_videos_per_word)
        if len(video_data) == 0:
            raise ValueError("No video data available for training!")
        
        dataset = WLASLVideoDataset(video_data)
        num_classes = len(dataset.label_encoder.classes_)
        
        logger.info(f"Dataset created with {len(dataset)} videos")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Classes: {list(dataset.label_encoder.classes_)}")
        
        # Split data
        if len(video_data) > 1:
            train_indices, val_indices = train_test_split(
                range(len(dataset)), test_size=test_size, 
                stratify=dataset.encoded_labels, random_state=42
            )
        else:
            train_indices = val_indices = list(range(len(dataset)))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        model = Simple3DCNN(num_classes=num_classes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        label_mapping = {i: label for i, label in enumerate(dataset.label_encoder.classes_)}
        with open('wlasl_label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                frames = batch['frames'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*train_correct/train_total:.2f}%"
                })
            
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc="Validation")
                for batch in pbar_val:
                    frames = batch['frames'].to(self.device)
                    labels_batch = batch['labels'].to(self.device)
                    outputs = model(frames)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_loss': avg_train_loss,
                    'num_classes': num_classes,
                    'label_encoder_classes': list(dataset.label_encoder.classes_)
                }, 'best_wlasl_model.pth')
                logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            
            scheduler.step()
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        torch.save(model.state_dict(), 'final_wlasl_model.pth')
        
        history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
            'classes': list(dataset.label_encoder.classes_)
        }
        with open('wlasl_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return model, history

def text_to_speech(text, language='en'):
    """Convert text to speech using Dwani API"""
    try:
        if not text:
            logger.error("Empty text provided to text_to_speech")
            return None

        # Generate a unique filename based on text content and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        filename = f'speech_{timestamp}_{text_hash}.mp3'
        output_file = os.path.join(app.config['AUDIO_FOLDER'], filename)
        
        # Check cache first
        cache_key = f"{text}_{language}"
        if cache_key in speech_cache:
            logger.info(f"Using cached audio for text: {text[:50]}...")
            return f'/download/{os.path.basename(speech_cache[cache_key])}'
        
        logger.info(f"Generating speech for text: {text[:50]}...")
        logger.info(f"Using language: {language}")
        
        try:
            audio_data = dwani.Audio.speech(
                input=text,
                response_format="mp3",
                language=language
            )
        except Exception as dwani_error:
            logger.error(f"Dwani API error: {str(dwani_error)}")
            return None
        
        if not audio_data:
            logger.error("No audio data received from Dwani API")
            return None
        
        try:
            with open(output_file, 'wb') as file:
                file.write(audio_data)
            logger.info(f"Successfully saved audio to: {output_file}")
        except Exception as file_error:
            logger.error(f"Error saving audio file: {str(file_error)}")
            return None
        
        speech_cache[cache_key] = output_file
        return f'/download/{filename}'
    except Exception as e:
        logger.error(f"Unexpected error in text_to_speech: {str(e)}")
        return None

@app.route('/api/save-video', methods=['POST'])
def save_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Save video to Videos folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f'sign_recording_{timestamp}.mp4')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure the Videos directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the video file
        video_file.save(filepath)
        logger.info(f"Video saved to: {filepath}")
        
        # Verify the video file exists and is readable
        if not os.path.exists(filepath):
            logger.error(f"Video file not found after saving: {filepath}")
            return jsonify({
                'success': False,
                'error': 'Failed to save video file'
            }), 500
            
        # Verify video can be opened
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {filepath}")
            return jsonify({
                'success': False,
                'error': 'Invalid video file'
            }), 500
        cap.release()
        
        # Send video to model for prediction using ResNet3D
        logger.info(f"Starting prediction for video: {filepath}")
        predicted_text = predict_sign(filepath)
        
        if not predicted_text:
            logger.error("Failed to predict sign language from video")
            return jsonify({
                'success': False,
                'error': 'Failed to predict sign language from video'
            }), 500

        logger.info(f"Predicted text: {predicted_text}")
        
        # Convert prediction to speech using Dwani
        audio_file = text_to_speech(predicted_text)
        if not audio_file:
            logger.error("Failed to generate speech from prediction")
            return jsonify({
                'success': False,
                'error': 'Failed to generate speech from prediction'
            }), 500
        
        logger.info(f"Generated audio file: {audio_file}")
        
        # Return success response with all necessary information
        return jsonify({
            'success': True,
            'path': filepath,
            'message': 'Video processed successfully',
            'filename': filename,
            'recognized_text': predicted_text,
            'audio_file': audio_file
        })
        
    except Exception as e:
        logger.error(f"Error in save_video: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak_text():
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        audio_file = text_to_speech(text, language)
        if not audio_file:
            return jsonify({'success': False, 'error': 'Failed to generate speech'})
        
        return jsonify({
            'success': True,
            'audio_file': audio_file
        })
    except Exception as e:
        logger.error(f"Error in speak_text: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test-dwani', methods=['GET'])
def test_dwani():
    try:
        test_text = "Hello, this is a test."
        audio_file = text_to_speech(test_text)
        if audio_file:
            return jsonify({
                'success': True,
                'message': 'Dwani API is working',
                'audio_file': audio_file
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate speech'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Check in the audio folder first
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        if os.path.exists(audio_path):
            return send_file(audio_path, mimetype='audio/mpeg')
        
        # If not found in audio folder, check other possible locations
        possible_paths = [
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            os.path.join(app.config['SIGN_OUTPUT_FOLDER'], filename),
            os.path.join(app.config['COMBINED_OUTPUT_FOLDER'], filename)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path, mimetype='audio/mpeg')
        
        logger.error(f"File not found: {filename}")
        return jsonify({'success': False, 'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def main():
    """Main function to run WLASL training"""
    try:
        trainer = WLASLTrainer(data_dir="wlasl_data")
        print("WLASL Dataset Training")
        print("=" * 50)
        print("This will download and train on a subset of the WLASL dataset")
        print("Note: Full dataset is very large, so we'll use a subset")
        print()
        
        max_words = int(input("Number of words to download (default 20): ") or "20")
        max_videos_per_word = int(input("Max videos per word (default 10): ") or "10")
        epochs = int(input("Number of training epochs (default 15): ") or "15")
        batch_size = int(input("Batch size (default 2): ") or "2")
        
        print(f"\nTraining Configuration:")
        print(f"- Words: {max_words}")
        print(f"- Videos per word: {max_videos_per_word}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Estimated total videos: {max_words * max_videos_per_word}")
        print()
        
        model, history = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-4,
            test_size=0.2,
            max_words=max_words,
            max_videos_per_word=max_videos_per_word
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print("="*50)
        print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
        print(f"Number of classes: {history['num_classes']}")
        print(f"Classes trained: {', '.join(history['classes'][:10])}...")
        print("\nSaved files:")
        print("- best_wlasl_model.pth (best model)")
        print("- final_wlasl_model.pth (final model)")
        print("- wlasl_label_mapping.json (class labels)")
        print("- wlasl_training_history.json (training metrics)")
        print(f"- wlasl_data/ (downloaded videos)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
