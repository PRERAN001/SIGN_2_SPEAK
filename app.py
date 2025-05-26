from flask import Flask, request, jsonify, send_file, render_template_string,render_template
from flask_cors import CORS
import torch
import json
import os
import cv2
import numpy as np
from app import WLASLVideoDataset
import logging
import tempfile
import torch.nn.functional as F
from pathlib import Path
import shutil
from werkzeug.utils import secure_filename
from datetime import datetime
import sqlite3
import uuid
import hashlib
import sys
import io
import dwani
import os
from deep_translator import GoogleTranslator
import hashlib
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Dhwani API configuration
dwani.api_key = os.getenv("DWANI_API_KEY", "preran248@gmail.com_dwani")  
dwani.api_base = os.getenv("DWANI_API_BASE_URL", "https://dwani-dwani-api.hf.space")

# Cache for translations
translation_cache = {}

def translate_to_kannada(text):
    # Check cache first
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    if text_hash in translation_cache:
        return translation_cache[text_hash]
    
    try:
        translator = GoogleTranslator(source='en', target='kn')
        result = translator.translate(text)
        translation_cache[text_hash] = result
        return result
    except Exception as e:
        return f"Error during translation: {str(e)}"
def text_to_speech(kannada_text):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_hash = hashlib.md5(kannada_text.encode('utf-8')).hexdigest()[:8]
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"output_{timestamp}_{text_hash}.mp3")

        # Generate speech using Dhwani TTS
        try:
            audio_data = dwani.Audio.speech(
                input=kannada_text,
                response_format="mp3"
            )
        except Exception as api_error:
            logger.error(f"Dhwani API error: {str(api_error)}")
            return f"Error during TTS: API call failed - {str(api_error)}"

        # Validate audio_data
        if not audio_data or not isinstance(audio_data, bytes):
            logger.error(f"Invalid audio data received for text: {kannada_text}")
            return f"Error during TTS: Invalid audio data received"

        if len(audio_data) < 100:  # Minimum size check
            logger.error(f"Audio data too small ({len(audio_data)} bytes) for text: {kannada_text}")
            return f"Error during TTS: Audio data too small"

        # Ensure output directory exists
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

        # Save audio with proper error handling
        try:
            with open(output_file, 'wb') as file:
                file.write(audio_data)
                file.flush()  # Ensure data is written
                os.fsync(file.fileno())  # Ensure data is synced to disk
        except IOError as io_error:
            logger.error(f"Failed to write audio file: {str(io_error)}")
            return f"Error during TTS: Failed to save audio file - {str(io_error)}"
        
        # Verify file was written correctly
        if not os.path.exists(output_file):
            logger.error(f"Audio file was not created: {output_file}")
            return "Error during TTS: File was not created"

        file_size = os.path.getsize(output_file)
        if file_size < 100:  # Adjust threshold as needed
            logger.error(f"Generated audio file is too small ({file_size} bytes): {output_file}")
            os.remove(output_file)
            return "Error during TTS: Generated audio file is too small"

        logger.info(f"Audio file generated successfully: {output_file} ({file_size} bytes)")
        return os.path.basename(output_file)  # Return just the filename

    except Exception as e:
        logger.error(f"Unexpected error during TTS: {str(e)}")
        return f"Error during TTS: {str(e)}"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['COMMUNITY_UPLOADS'] = 'community_uploads'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['COMMUNITY_UPLOADS'], exist_ok=True)

# Global variables for model and dataset
model = None
dataset = None
device = None

# Initialize database
def init_database():
    """Initialize SQLite database for community features"""
    conn = sqlite3.connect('asl_community.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        profile_pic TEXT,
        bio TEXT,
        learning_level TEXT DEFAULT 'beginner',
        experience_points INTEGER DEFAULT 0,
        level INTEGER DEFAULT 1
    )
    ''')
    
    # Posts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT,
        video_path TEXT,
        sign_word TEXT,
        difficulty_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        likes INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Comments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id TEXT PRIMARY KEY,
        post_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts (id),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Likes table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS likes (
        id TEXT PRIMARY KEY,
        post_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(post_id, user_id),
        FOREIGN KEY (post_id) REFERENCES posts (id),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Learning progress table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS learning_progress (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        sign_word TEXT NOT NULL,
        practice_count INTEGER DEFAULT 0,
        accuracy_score REAL DEFAULT 0.0,
        last_practiced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, sign_word),
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Achievements table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS achievements (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        xp_reward INTEGER NOT NULL,
        icon TEXT NOT NULL
    )
    ''')
    
    # User achievements table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_achievements (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        achievement_id TEXT NOT NULL,
        earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (achievement_id) REFERENCES achievements (id)
    )
    ''')
    
    # Learning milestones table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS learning_milestones (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        milestone_type TEXT NOT NULL,
        milestone_value INTEGER NOT NULL,
        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Insert default achievements if they don't exist
    default_achievements = [
        ('first_sign', 'First Sign', 'Successfully recognized your first sign', 50, '🎯'),
        ('practice_master', 'Practice Master', 'Practiced 10 different signs', 100, '🏆'),
        ('community_contributor', 'Community Contributor', 'Created your first post', 75, '🌟'),
        ('helpful_member', 'Helpful Member', 'Received 5 likes on your posts', 150, '💫'),
        ('sign_expert', 'Sign Expert', 'Achieved 90% accuracy on 5 signs', 200, '👑'),
        ('learning_streak', 'Learning Streak', 'Practiced for 7 consecutive days', 300, '🔥'),
        ('video_creator', 'Video Creator', 'Uploaded your first sign language video', 125, '🎥'),
        ('comment_enthusiast', 'Comment Enthusiast', 'Left 10 helpful comments', 100, '💬'),
        ('beginner_master', 'Beginner Master', 'Completed all beginner-level signs', 250, '🎓'),
        ('intermediate_learner', 'Intermediate Learner', 'Reached intermediate level', 400, '📚')
    ]
    
    cursor.execute('SELECT COUNT(*) FROM achievements')
    if cursor.fetchone()[0] == 0:
        for achievement in default_achievements:
            cursor.execute('''
            INSERT INTO achievements (id, name, description, xp_reward, icon)
            VALUES (?, ?, ?, ?, ?)
            ''', achievement)
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect('asl_community.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_resnet3d(num_classes):
    """Create ResNet3D model"""
    import torchvision
    model = torchvision.models.video.r3d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def preprocess_video(video_path, max_frames=16, img_size=224):
    """Preprocess video for model input"""
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
        raise ValueError("No frames found in the video.")
    
    # Uniformly sample max_frames from the video
    if len(frames) >= max_frames:
        idxs = np.linspace(0, len(frames)-1, max_frames).astype(int)
        frames = [frames[i] for i in idxs]
    else:
        while len(frames) < max_frames:
            frames.append(frames[-1])
    
    # Stack frames and transpose to [channels, num_frames, height, width]
    frames = np.stack(frames)  # Shape: [num_frames, height, width, channels]
    frames = frames.transpose(3, 0, 1, 2)  # Shape: [channels, num_frames, height, width]
    frames = torch.tensor(frames, dtype=torch.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    return frames

def create_sign_video(input_text, base_dir, output_filename="output_custom.mp4", 
                     default_resolution=(224, 224), placeholder_color=(0, 0, 0)):
    """
    Create a combined sign language video from input text by concatenating videos for each word.
    """
    temp_dir = None
    try:
        base_dir = Path(base_dir)
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        output_path = output_dir / output_filename

        temp_dir = Path(tempfile.mkdtemp())

        if not base_dir.exists():
            logger.error(f"Base directory does not exist: {base_dir}")
            return False, f"Base directory does not exist: {base_dir}"

        words = input_text.lower().strip().split()
        if not words:
            logger.error("No valid words provided in input text")
            return False, "No valid words provided in input text"

        # Collect video paths
        video_paths = []
        missing_words = []
        for word in words:
            word_folder = base_dir / word
            if word_folder.exists():
                mp4_files = list(word_folder.glob("*.mp4"))
                if mp4_files:
                    video_paths.append((word, mp4_files[0]))
                    logger.info(f"Using video for '{word}': {mp4_files[0]}")
                else:
                    logger.warning(f"No .mp4 files found in folder: {word}")
                    video_paths.append((word, None))
                    missing_words.append(word)
            else:
                logger.warning(f"Folder not found for word: {word}")
                video_paths.append((word, None))
                missing_words.append(word)

        # Initialize video writer parameters
        width, height = default_resolution
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        frames_written = 0

        # Process videos
        for word, video_path in video_paths:
            if video_path is None:
                logger.info(f"Generating 1-second placeholder for '{word}'")
                placeholder_frame = np.full((height, width, 3), placeholder_color, dtype=np.uint8)
                if out is None:
                    temp_output = temp_dir / "temp_output.mp4"
                    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
                    if not out.isOpened():
                        logger.error(f"Failed to initialize video writer")
                        return False, "Failed to initialize video writer"
                for _ in range(int(fps)):
                    out.write(placeholder_frame)
                    frames_written += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video for '{word}': {video_path}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            if out is None:
                fps = video_fps
                temp_output = temp_dir / "temp_output.mp4"
                out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
                if not out.isOpened():
                    logger.error(f"Failed to initialize video writer")
                    cap.release()
                    return False, "Failed to initialize video writer"

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, default_resolution, interpolation=cv2.INTER_AREA)
                out.write(frame)
                frames_written += 1

            cap.release()
            logger.info(f"Processed {word}")

        if out is not None:
            out.release()
            if frames_written > 0:
                shutil.move(str(temp_output), str(output_path))
                logger.info(f"Final video saved at: {output_path} with {frames_written} frames")
                return True, str(output_path), missing_words
            else:
                logger.warning("No frames written. Check video files.")
                if temp_output.exists():
                    temp_output.unlink()
                return False, "No frames written"
        else:
            logger.warning("No valid videos processed. Output not created.")
            return False, "No valid videos processed"

    except Exception as e:
        logger.error(f"Error in create_sign_video: {str(e)}")
        return False, str(e)
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)

def initialize_model():
    """Initialize the recognition model and dataset"""
    global model, dataset, device
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load dataset info
        with open("wlasl_data/download_info.json", "r") as f:
            video_data = json.load(f)["data"]
        
        dataset = WLASLVideoDataset(video_data)
        num_classes = len(dataset.label_encoder.classes_)
        logger.info(f"Number of classes: {num_classes}")
        
        # Load model
        model = get_resnet3d(num_classes).to(device)
        model.load_state_dict(torch.load("best_resnet3d_wlasl.pth", map_location=device))
        model.eval()
        
        logger.info("Model initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

@app.route('/')
def index():
    """Serve the main HTML page"""
    html_content = '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Recognition Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow-x: hidden;
            line-height: 1.6;
            color: #333;
        }

        /* Loading Animation Styles */
        #loadingScreen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            transition: opacity 0.5s ease-out;
        }

        .cube-container {
            width: 200px;
            height: 200px;
            perspective: 800px;
            margin-bottom: 30px;
        }

        .cube {
            position: relative;
            width: 100%;
            height: 100%;
            transform-style: preserve-3d;
            animation: rotateCube 6s infinite linear;
        }

        .cube-face {
            position: absolute;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 60px;
            font-weight: bold;
            color: white;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(5px);
        }

        .cube-face.front  { transform: rotateY(0deg) translateZ(100px); }
        .cube-face.back   { transform: rotateY(180deg) translateZ(100px); }
        .cube-face.right  { transform: rotateY(90deg) translateZ(100px); }
        .cube-face.left   { transform: rotateY(-90deg) translateZ(100px); }
        .cube-face.top    { transform: rotateX(90deg) translateZ(100px); }
        .cube-face.bottom { transform: rotateX(-90deg) translateZ(100px); }

        @keyframes rotateCube {
            0% { transform: rotateX(0deg) rotateY(0deg); }
            100% { transform: rotateX(360deg) rotateY(360deg); }
        }

        .loading-text {
            color: white;
            font-size: 2rem;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
            animation: fadeInText 1s ease-in forwards;
            opacity: 0;
        }

        .loading-subtitle {
            color: rgba(255,255,255,0.8);
            font-size: 1.2rem;
            margin-top: 10px;
            text-align: center;
            animation: fadeInText 1.5s ease-in forwards;
            opacity: 0;
        }

        .progress-circle {
            width: 60px;
            height: 60px;
            margin-top: 20px;
            position: relative;
        }

        .progress-circle svg {
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
        }

        .progress-circle circle {
            fill: none;
            stroke: rgba(255, 255, 255, 0.3);
            stroke-width: 6;
        }

        .progress-circle .progress {
            stroke: white;
            stroke-dasharray: 163.36;
            stroke-dashoffset: 163.36;
            animation: progress 5s linear forwards;
        }

        @keyframes fadeInText {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes progress {
            0% { stroke-dashoffset: 163.36; }
            100% { stroke-dashoffset: 0; }
        }

        /* Main Content Styles */
        #mainContent {
            opacity: 0;
            transition: opacity 1s ease-in;
            min-height: 100vh;
            background: #f8f9fa;
        }

        #mainContent.show {
            opacity: 1;
        }

        /* Enhanced Navigation */
        .navbar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 0;
            color: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .navbar::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
            position: relative;
            z-index: 1;
        }

        .logo {
            font-size: 2.2rem;
            font-weight: bold;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
            animation: logoGlow 3s ease-in-out infinite alternate;
        }

        @keyframes logoGlow {
            from { text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
            to { text-shadow: 0 2px 20px rgba(255,255,255,0.5); }
        }

        .nav-links {
            display: flex;
            gap: 2rem;
        }

        .nav-link {
            color: white;
            text-decoration: none;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .nav-link::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.2);
            transition: left 0.3s ease;
        }

        .nav-link:hover::before {
            left: 0;
        }

        .nav-link:hover, .nav-link.active {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        /* Enhanced Main Content */
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        .container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            margin: 30px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .container:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }

        .section {
            margin: 30px 0;
        }

        .section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 2rem;
            position: relative;
            padding-bottom: 10px;
        }

        .section h2::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 50px;
            height: 3px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }

        /* Enhanced Forms and Inputs */
        .upload-area {
            border: 3px dashed #007bff;
            padding: 50px;
            text-align: center;
            border-radius: 20px;
            background: linear-gradient(135deg, #f8f9ff 0%, #e3f2fd 100%);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .upload-area::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(0,123,255,0.1);
            border-radius: 50%;
            transition: all 0.3s ease;
            transform: translate(-50%, -50%);
        }

        .upload-area:hover::before {
            width: 300px;
            height: 300px;
        }

        .upload-area:hover {
            border-color: #0056b3;
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            transform: scale(1.02);
        }

        .form-group {
            margin: 20px 0;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
            font-size: 1.1rem;
        }

        .form-control {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e1e1e1;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #fafafa;
        }

        .form-control:focus {
            outline: none;
            border-color: #007bff;
            background: white;
            box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
            transform: translateY(-2px);
        }

        /* Enhanced Buttons */
        .btn {
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            margin: 8px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,123,255,0.4);
            background: linear-gradient(135deg, #0056b3, #004085);
        }

        .btn:active {
            transform: translateY(-1px);
        }

        /* Enhanced Results and Messages */
        .result {
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }

        .result::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }

        .error {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            color: #721c24;
            border-left: 4px solid #dc3545;
        }

        .success {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724;
            border-left: 4px solid #28a745;
        }

        .info {
            background: linear-gradient(135deg, #d1ecf1, #bee5eb);
            color: #0c5460;
            border-left: 4px solid #17a2b8;
        }

        .loading {
            text-align: center;
            margin: 25px;
            padding: 25px;
            font-size: 1.2rem;
            color: #666;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .nav-container {
                flex-direction: column;
                gap: 1rem;
            }

            .nav-links {
                flex-wrap: wrap;
                justify-content: center;
            }

            .main-container {
                padding: 20px 10px;
            }

            .container {
                padding: 25px;
                margin: 15px 0;
            }

            .loading-text {
                font-size: 1.5rem;
            }

            .loading-subtitle {
                font-size: 1rem;
            }

            .cube-container {
                width: 150px;
                height: 150px;
            }

            .cube-face {
                width: 150px;
                height: 150px;
                font-size: 40px;
            }

            .cube-face.front  { transform: rotateY(0deg) translateZ(75px); }
            .cube-face.back   { transform: rotateY(180deg) translateZ(75px); }
            .cube-face.right  { transform: rotateY(90deg) translateZ(75px); }
            .cube-face.left   { transform: rotateY(-90deg) translateZ(75px); }
            .cube-face.top    { transform: rotateX(90deg) translateZ(75px); }
            .cube-face.bottom { transform: rotateX(-90deg) translateZ(75px); }
        }

        /* Accessibility Improvements */
        :focus-visible {
            outline: 3px solid #007bff;
            outline-offset: 2px;
        }

        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            border: 0;
        }

        /* Performance Optimizations */
        .container, .btn, .nav-link, .upload-area {
            will-change: transform, box-shadow;
        }

        /* Gamified Learning Styles */
        .learning-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .video-card {
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            transition: transform 0.3s ease;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .video-card:hover {
            transform: translateY(-5px);
        }

        .video-thumbnail {
            width: 100%;
            height: 180px;
            object-fit: cover;
        }

        .video-info {
            padding: 15px;
        }

        .video-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }

        .video-description {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 10px;
        }

        .video-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: #888;
        }

        .difficulty-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .difficulty-beginner { background: #d4edda; color: #155724; }
        .difficulty-intermediate { background: #fff3cd; color: #856404; }
        .difficulty-advanced { background: #f8d7da; color: #721c24; }

        .progress-bar {
            width: 100%;
            height: 6px;
            background: #e9ecef;
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }

        /* Dark Mode Support */
        @media (prefers-color-scheme: dark) {
            body {
                background: linear-gradient(135deg, #2a2a6e 0%, #3a1a5e 100%);
                color: #ddd;
            }

            #mainContent {
                background: #1a1a1a;
            }

            .container {
                background: #2a2a2a;
                color: #ddd;
            }

            .form-control {
                background: #333;
                color: #ddd;
                border-color: #555;
            }

            .form-control:focus {
                background: #444;
                border-color: #007bff;
            }

            .video-card {
                background: #333;
            }

            .video-title {
                color: #ddd;
            }

            .video-description {
                color: #aaa;
            }

            .video-meta {
                color: #999;
            }
        }

        /* Camera Recording Styles */
        .camera-container {
            position: relative;
            width: 100%;
            max-width: 640px;
            margin: 20px auto;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }

        #videoElement {
            width: 100%;
            height: auto;
            display: block;
        }

        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 15px 0;
        }

        .record-button {
            background: #dc3545;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .record-button:hover {
            background: #c82333;
        }

        .record-button.recording {
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        .camera-status {
            text-align: center;
            margin: 10px 0;
            color: #666;
        }
    </style>
</head>
<body>
    <!-- Loading Screen -->
    <div id="loadingScreen">
        <div class="cube-container">
            <div class="cube">
                <div class="cube-face front">ASL</div>
                <div class="cube-face back">ASL</div>
                <div class="cube-face right">ASL</div>
                <div class="cube-face left">ASL</div>
                <div class="cube-face top">ASL</div>
                <div class="cube-face bottom">ASL</div>
            </div>
        </div>
        <div class="loading-text">ASL Recognition Platform</div>
        <div class="loading-subtitle">Initializing AI Models...</div>
        <div class="progress-circle">
            <svg>
                <circle cx="30" cy="30" r="26"></circle>
                <circle class="progress" cx="30" cy="30" r="26"></circle>
            </svg>
        </div>
    </div>

    <!-- Main Content -->
    <div id="mainContent">
        <!-- Navigation -->
        <nav class="navbar">
            <div class="nav-container">
                <div class="logo">ASL Recognition</div>
                <div class="nav-links">
                    <a href="/" class="nav-link active">Recognition</a>
                    <a href="/community" class="nav-link">Community</a>
                    <a href="/learning" class="nav-link">Learning</a>
                </div>
            </div>
        </nav>

        <div class="main-container">
            <!-- Camera Recording Section -->
            <div class="container">
                <h2>🎥 Camera Recognition</h2>
                <p>Record your ASL gestures using your camera</p>
                <div class="camera-container">
                    <video id="videoElement" autoplay playsinline></video>
                </div>
                <div class="camera-controls">
                    <button id="startCamera" class="btn">Start Camera</button>
                    <button id="recordButton" class="record-button" disabled>Record</button>
                    <button id="stopCamera" class="btn" disabled>Stop Camera</button>
                </div>
                <div class="camera-status" id="cameraStatus">Camera is off</div>
                <div id="recordingResult" class="result" style="display: none;"></div>
                <div id="recordingLoading" class="loading" style="display: none;">🔄 Processing...</div>
            </div>

            <!-- Recognition Section -->
            <div class="container">
                <h2>📹 Video Recognition</h2>
                <p>Upload a video to recognize ASL gestures</p>
                <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                    <p>Click here to select a video file</p>
                    <input type="file" id="videoInput" accept="video/*" style="display: none;">
                </div>
                <div id="recognitionResult" class="result" style="display: none;"></div>
                <div id="recognitionLoading" class="loading" style="display: none;">🔄 Processing...</div>
            </div>

            <!-- Generation Section -->
            <div class="container">
                <h2>✍️ Text to Sign Video</h2>
                <p>Enter text to generate a sign language video</p>
                <div class="form-group">
                    <input type="text" id="textInput" class="form-control" placeholder="Enter text (e.g., hello world)">
                </div>
                <button class="btn" onclick="generateVideo()">Generate Video</button>
                <div id="generationResult" class="result" style="display: none;"></div>
                <div id="generationLoading" class="loading" style="display: none;">🎬 Generating video...</div>
            </div>

            <!-- System Status -->
            <div class="container">
                <h2>⚙️ System Status</h2>
                <button class="btn" onclick="checkHealth()">Check Status</button>
                <div id="healthResult" class="result" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        // Loading Animation Control
        function initLoadingAnimation() {
            // Animation is handled purely by CSS
            setTimeout(hideLoadingScreen, 5000); // 5 seconds
        }

        function hideLoadingScreen() {
            const loadingScreen = document.getElementById('loadingScreen');
            const mainContent = document.getElementById('mainContent');

            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.style.display = 'none';
                mainContent.classList.add('show');
            }, 500);
        }

        // Initialize animation when page loads
        window.addEventListener('load', initLoadingAnimation);

        // Video Recognition
        document.getElementById('videoInput').addEventListener('change', async function(e) {
            const file = e.target.files[0];
            if (!file) return;

            document.getElementById('recognitionLoading').style.display = 'block';
            document.getElementById('recognitionResult').style.display = 'none';

            const formData = new FormData();
            formData.append('video', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    let resultHTML = `
                        <h3>Recognition Result:</h3>
                        <p><strong>Predicted Sign:</strong> ${result.prediction}</p>
                        <p><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</p>
                        <p><strong>Kannada Translation:</strong> ${result.kannada_translation}</p>
                    `;
                    if (result.audio_file) {
                        resultHTML += `
                            <p><strong>Listen to Translation:</strong></p>
                            <audio controls>
                                <source src="/download/${result.audio_file}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                        `;
                    }
                    document.getElementById('recognitionResult').innerHTML = resultHTML;
                    document.getElementById('recognitionResult').className = 'result success';
                } else {
                    document.getElementById('recognitionResult').innerHTML = `<p>Error: ${result.error}</p>`;
                    document.getElementById('recognitionResult').className = 'result error';
                }
                document.getElementById('recognitionResult').style.display = 'block';
            } catch (error) {
                document.getElementById('recognitionResult').innerHTML = `<p>Error: ${error.message}</p>`;
                document.getElementById('recognitionResult').className = 'result error';
                document.getElementById('recognitionResult').style.display = 'block';
            } finally {
                document.getElementById('recognitionLoading').style.display = 'none';
            }
        });

        // Video Generation
        async function generateVideo() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                alert('Please enter some text');
                return;
            }

            document.getElementById('generationLoading').style.display = 'block';
            document.getElementById('generationResult').style.display = 'none';

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const result = await response.json();

                if (response.ok) {
                    document.getElementById('generationResult').innerHTML = 
                        `<h3>Video Generated Successfully!</h3>
                         <p><strong>Filename:</strong> ${result.filename}</p>
                         <p><strong>Words processed:</strong> ${result.words_processed}</p>
                         ${result.missing_words && result.missing_words.length > 0 ? 
                           `<p><strong>Missing words:</strong> ${result.missing_words.join(', ')}</p>` : ''}
                         <a href="/download/${result.filename}" class="btn">Download Video</a>`;
                    document.getElementById('generationResult').className = 'result success';
                } else {
                    document.getElementById('generationResult').innerHTML = `<p>Error: ${result.error}</p>`;
                    document.getElementById('generationResult').className = 'result error';
                }
                document.getElementById('generationResult').style.display = 'block';
            } catch (error) {
                document.getElementById('generationResult').innerHTML = `<p>Error: ${error.message}</p>`;
                document.getElementById('generationResult').className = 'result error';
                document.getElementById('generationResult').style.display = 'block';
            } finally {
                document.getElementById('generationLoading').style.display = 'none';
            }
        }

        // System Health
        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const result = await response.json();

                document.getElementById('healthResult').innerHTML = 
                    `<h3>System Status:</h3>
                     <p><strong>Status:</strong> ${result.status}</p>
                     <p><strong>Model Loaded:</strong> ${result.model_loaded ? 'Yes' : 'No'}</p>
                     <p><strong>Device:</strong> ${result.device || 'N/A'}</p>`;
                document.getElementById('healthResult').className = 'result success';
                document.getElementById('healthResult').style.display = 'block';
            } catch (error) {
                document.getElementById('healthResult').innerHTML = `<p>Error: ${error.message}</p>`;
                document.getElementById('healthResult').className = 'result error';
                document.getElementById('healthResult').style.display = 'block';
            }
        }

        // Camera Recording Functionality
        let mediaRecorder;
        let recordedChunks = [];
        let stream;
        let isRecording = false;

        const videoElement = document.getElementById('videoElement');
        const startCameraButton = document.getElementById('startCamera');
        const recordButton = document.getElementById('recordButton');
        const stopCameraButton = document.getElementById('stopCamera');
        const cameraStatus = document.getElementById('cameraStatus');
        const recordingResult = document.getElementById('recordingResult');
        const recordingLoading = document.getElementById('recordingLoading');

        startCameraButton.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    },
                    audio: false 
                });
                videoElement.srcObject = stream;
                startCameraButton.disabled = true;
                recordButton.disabled = false;
                stopCameraButton.disabled = false;
                cameraStatus.textContent = 'Camera is on';
            } catch (err) {
                console.error('Error accessing camera:', err);
                cameraStatus.textContent = 'Error accessing camera';
            }
        });

        recordButton.addEventListener('click', () => {
            if (!isRecording) {
                // Start recording
                recordedChunks = [];
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    const blob = new Blob(recordedChunks, { type: 'video/webm' });
                    const formData = new FormData();
                    formData.append('video', blob, 'recording.webm');

                    recordingLoading.style.display = 'block';
                    recordingResult.style.display = 'none';

                    try {
                        const response = await fetch('/camera-record', {
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();

                        if (response.ok) {
                            let resultHTML = `
                                <h3>Recognition Result:</h3>
                                <p><strong>Predicted Sign:</strong> ${result.prediction}</p>
                                <p><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</p>
                                <p><strong>Kannada Translation:</strong> ${result.kannada_translation}</p>
                            `;
                            if (result.audio_file) {
                                resultHTML += `
                                    <p><strong>Listen to Translation:</strong></p>
                                    <audio controls>
                                        <source src="/download/${result.audio_file}" type="audio/mp3">
                                        Your browser does not support the audio element.
                                    </audio>
                                `;
                            }
                            recordingResult.innerHTML = resultHTML;
                            recordingResult.className = 'result success';
                        } else {
                            recordingResult.innerHTML = `<p>Error: ${result.error}</p>`;
                            recordingResult.className = 'result error';
                        }
                        recordingResult.style.display = 'block';
                    } catch (error) {
                        recordingResult.innerHTML = `<p>Error: ${error.message}</p>`;
                        recordingResult.className = 'result error';
                        recordingResult.style.display = 'block';
                    } finally {
                        recordingLoading.style.display = 'none';
                    }
                };

                mediaRecorder.start();
                isRecording = true;
                recordButton.textContent = 'Stop Recording';
                recordButton.classList.add('recording');
                cameraStatus.textContent = 'Recording...';
            } else {
                // Stop recording
                mediaRecorder.stop();
                isRecording = false;
                recordButton.textContent = 'Record';
                recordButton.classList.remove('recording');
                cameraStatus.textContent = 'Processing recording...';
            }
        });

        stopCameraButton.addEventListener('click', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                videoElement.srcObject = null;
                startCameraButton.disabled = false;
                recordButton.disabled = true;
                stopCameraButton.disabled = true;
                cameraStatus.textContent = 'Camera is off';
            }
        });
    </script>
</body>
</html>
    '''
    return render_template_string(html_content)

@app.route('/community')
def community():
    """Serve the community page"""
    return render_template('community.html')

@app.route('/learning')
def learning():
    """Serve the learning page"""
    return render_template('gameified_learning.html')

@app.route('/api/quiz-questions', methods=['GET'])
def api_quiz_questions():
    """Get quiz questions"""
    try:
        questions = [
            {
                'id': '1',
                'video': '/gamified_video/hello.mp4',
                'question': 'What does this sign mean?',
                'options': ['Hello', 'Goodbye', 'Thank you', 'Please'],
                'correctAnswer': 'Hello'
            },
            {
                'id': '2',
                'video': '/gamified_video/dance.mp4',
                'question': 'What does this sign mean?',
                'options': ['Dance', 'No', 'Maybe', 'Later'],
                'correctAnswer': 'Dance'
            },
            {
                'id': '3',
                'video': '/gamified_video/happy.mp4',
                'question': 'What does this sign mean?',
                'options': ['Happy', 'No', 'Maybe', 'Later'],
                'correctAnswer': 'Happy'
            },
            {
                'id': '4',
                'video': '/gamified_video/more.mp4',
                'question': 'What does this sign mean?',
                'options': ['Yes', 'No', 'More', 'Later'],
                'correctAnswer': 'More'
            },
            {
                'id': '5',
                'video': '/gamified_video/no.mp4',
                'question': 'What does this sign mean?',
                'options': ['Yes', 'No', 'Maybe', 'Later'],
                'correctAnswer': 'No'
            }
        ]
        return jsonify({'questions': questions})
    except Exception as e:
        logger.error(f"Error fetching quiz questions: {str(e)}")
        return jsonify({'error': 'Failed to fetch quiz questions'}), 500

@app.route('/api/submit-quiz', methods=['POST'])
def api_submit_quiz():
    """Submit quiz answers and update progress"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        answers = data.get('answers', [])
        
        if not username or not answers:
            return jsonify({'error': 'Username and answers are required'}), 400
        
        # Calculate score
        correct_answers = 0
        for answer in answers:
            if answer.get('isCorrect'):
                correct_answers += 1
        
        score = (correct_answers / len(answers)) * 100
        
        # Update learning progress
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get or create user
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            user_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO users (id, username, email, password_hash)
            VALUES (?, ?, ?, ?)
            ''', (user_id, username, f"{username}@temp.com", "no_password"))
        else:
            user_id = user['id']
        
        # Update progress for each sign word
        for answer in answers:
            cursor.execute('''
            INSERT INTO learning_progress (id, user_id, sign_word, practice_count, accuracy_score)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(user_id, sign_word) DO UPDATE SET
                practice_count = practice_count + 1,
                accuracy_score = (accuracy_score * practice_count + ?) / (practice_count + 1),
                last_practiced = CURRENT_TIMESTAMP
            ''', (str(uuid.uuid4()), user_id, answer['signWord'], score, score))
        
        # Check for achievements
        check_achievements(user_id)
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Quiz submitted successfully',
            'score': score,
            'correctAnswers': correct_answers,
            'totalQuestions': len(answers)
        })
        
    except Exception as e:
        logger.error(f"Error submitting quiz: {str(e)}")
        return jsonify({'error': 'Failed to submit quiz'}), 500

@app.route('/api/learning-videos', methods=['GET'])
def api_learning_videos():
    """Get learning videos"""
    try:
        # Sample video data - replace with actual database query
        videos = [
            {
                'id': '1',
                'title': 'Basic ASL Alphabet',
                'description': 'Learn the ASL alphabet with clear demonstrations',
                'thumbnail': '/static/thumbnails/alphabet.jpg',
                'difficulty': 'beginner',
                'duration': 10,
                'progress': 0
            },
            {
                'id': '2',
                'title': 'Common Greetings',
                'description': 'Master essential ASL greetings and introductions',
                'thumbnail': '/static/thumbnails/greetings.jpg',
                'difficulty': 'beginner',
                'duration': 15,
                'progress': 0
            },
            {
                'id': '3',
                'title': 'Numbers 1-20',
                'description': 'Learn to sign numbers from 1 to 20',
                'thumbnail': '/static/thumbnails/numbers.jpg',
                'difficulty': 'beginner',
                'duration': 12,
                'progress': 0
            },
            {
                'id': '4',
                'title': 'Basic Conversations',
                'description': 'Practice simple ASL conversations',
                'thumbnail': '/static/thumbnails/conversations.jpg',
                'difficulty': 'intermediate',
                'duration': 20,
                'progress': 0
            },
            {
                'id': '5',
                'title': 'Emotions and Feelings',
                'description': 'Express emotions and feelings in ASL',
                'thumbnail': '/static/thumbnails/emotions.jpg',
                'difficulty': 'intermediate',
                'duration': 18,
                'progress': 0
            },
            {
                'id': '6',
                'title': 'Advanced Grammar',
                'description': 'Master ASL grammar and sentence structure',
                'thumbnail': '/static/thumbnails/grammar.jpg',
                'difficulty': 'advanced',
                'duration': 25,
                'progress': 0
            }
        ]
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        logger.error(f"Error fetching learning videos: {str(e)}")
        return jsonify({'error': 'Failed to fetch learning videos'}), 500

# API Routes for Community Features
@app.route('/api/register', methods=['POST'])
def api_register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        learning_level = data.get('learning_level', 'beginner')
        
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Username or email already exists'}), 400
        
        # Create new user
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        
        cursor.execute('''
        INSERT INTO users (id, username, email, password_hash, learning_level)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, username, email, password_hash, learning_level))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User registered successfully'})
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', 
                      (username, password_hash))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return jsonify({
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'email': user['email'],
                    'learning_level': user['learning_level']
                }
            })
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/posts', methods=['GET', 'POST'])
def api_posts():
    """Get all posts or create a new post"""
    if request.method == 'GET':
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT p.*, u.username,
                   (SELECT COUNT(*) FROM likes l WHERE l.post_id = p.id) as likes
            FROM posts p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            ''')
            
            posts = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({'posts': posts})
            
        except Exception as e:
            logger.error(f"Error fetching posts: {str(e)}")
            return jsonify({'error': 'Failed to fetch posts'}), 500
    
    elif request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            if not username:
                return jsonify({'error': 'Username is required'}), 400
            
            title = request.form.get('title', '').strip()
            content = request.form.get('content', '').strip()
            sign_word = request.form.get('sign_word', '').strip()
            difficulty_level = request.form.get('difficulty_level', 'beginner')
            
            if not title:
                return jsonify({'error': 'Title is required'}), 400
            
            video_path = None
            if 'video' in request.files:
                video_file = request.files['video']
                if video_file.filename != '':
                    # Save video file
                    filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
                    video_path = os.path.join(app.config['COMMUNITY_UPLOADS'], filename)
                    video_file.save(video_path)
                    video_path = filename  # Store just the filename
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get or create user
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if not user:
                user_id = str(uuid.uuid4())
                cursor.execute('''
                INSERT INTO users (id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
                ''', (user_id, username, f"{username}@temp.com", "no_password"))
            else:
                user_id = user['id']
            
            post_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO posts (id, user_id, title, content, video_path, sign_word, difficulty_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (post_id, user_id, title, content, video_path, sign_word, difficulty_level))
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Post created successfully', 'post_id': post_id})
            
        except Exception as e:
            logger.error(f"Error creating post: {str(e)}")
            return jsonify({'error': 'Failed to create post'}), 500

@app.route('/api/posts/<post_id>/like', methods=['POST'])
def api_toggle_like(post_id):
    """Toggle like on a post"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get or create user
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            user_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO users (id, username, email, password_hash)
            VALUES (?, ?, ?, ?)
            ''', (user_id, username, f"{username}@temp.com", "no_password"))
        else:
            user_id = user['id']
        
        # Check if user already liked this post
        cursor.execute('SELECT id FROM likes WHERE post_id = ? AND user_id = ?', (post_id, user_id))
        existing_like = cursor.fetchone()
        
        if existing_like:
            # Remove like
            cursor.execute('DELETE FROM likes WHERE post_id = ? AND user_id = ?', (post_id, user_id))
        else:
            # Add like
            like_id = str(uuid.uuid4())
            cursor.execute('INSERT INTO likes (id, post_id, user_id) VALUES (?, ?, ?)', 
                          (like_id, post_id, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Like toggled successfully'})
        
    except Exception as e:
        logger.error(f"Error toggling like: {str(e)}")
        return jsonify({'error': 'Failed to toggle like'}), 500

@app.route('/api/posts/<post_id>/comments', methods=['GET', 'POST'])
def api_comments(post_id):
    """Get comments for a post or add a new comment"""
    if request.method == 'GET':
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT c.*, u.username
            FROM comments c
            JOIN users u ON c.user_id = u.id
            WHERE c.post_id = ?
            ORDER BY c.created_at ASC
            ''', (post_id,))
            
            comments = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({'comments': comments})
            
        except Exception as e:
            logger.error(f"Error fetching comments: {str(e)}")
            return jsonify({'error': 'Failed to fetch comments'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            content = data.get('content', '').strip()
            
            if not username:
                return jsonify({'error': 'Username is required'}), 400
            
            if not content:
                return jsonify({'error': 'Comment content is required'}), 400
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get or create user
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if not user:
                user_id = str(uuid.uuid4())
                cursor.execute('''
                INSERT INTO users (id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
                ''', (user_id, username, f"{username}@temp.com", "no_password"))
            else:
                user_id = user['id']
            
            comment_id = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO comments (id, post_id, user_id, content)
            VALUES (?, ?, ?, ?)
            ''', (comment_id, post_id, user_id, content))
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Comment added successfully'})
            
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            return jsonify({'error': 'Failed to add comment'}), 500

@app.route('/api/achievements', methods=['GET'])
def api_achievements():
    """Get user's achievements"""
    try:
        username = request.args.get('username', '').strip()
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Get user's achievements
        cursor.execute('''
        SELECT a.*, ua.earned_at
        FROM achievements a
        LEFT JOIN user_achievements ua ON a.id = ua.achievement_id AND ua.user_id = ?
        ORDER BY a.xp_reward DESC
        ''', (user['id'],))
        
        achievements = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'achievements': achievements})
        
    except Exception as e:
        logger.error(f"Error fetching achievements: {str(e)}")
        return jsonify({'error': 'Failed to fetch achievements'}), 500

@app.route('/api/progress', methods=['GET'])
def api_progress():
    """Get user's learning progress and statistics"""
    try:
        username = request.args.get('username', '').strip()
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Get learning progress
        cursor.execute('''
        SELECT sign_word, practice_count, accuracy_score, last_practiced
        FROM learning_progress
        WHERE user_id = ?
        ORDER BY last_practiced DESC
        ''', (user['id'],))
        
        progress = [dict(row) for row in cursor.fetchall()]
        
        # Get recent milestones
        cursor.execute('''
        SELECT milestone_type, milestone_value, completed_at
        FROM learning_milestones
        WHERE user_id = ?
        ORDER BY completed_at DESC
        LIMIT 5
        ''', (user['id'],))
        
        milestones = [dict(row) for row in cursor.fetchall()]
        
        # Calculate statistics
        total_practice_count = sum(p['practice_count'] for p in progress)
        average_accuracy = sum(p['accuracy_score'] for p in progress) / len(progress) if progress else 0
        
        conn.close()
        
        return jsonify({
            'user': {
                'username': user['username'],
                'level': user['level'],
                'experience_points': user['experience_points'],
                'learning_level': user['learning_level']
            },
            'progress': progress,
            'milestones': milestones,
            'statistics': {
                'total_practice_count': total_practice_count,
                'average_accuracy': round(average_accuracy, 2),
                'total_signs_learned': len(progress)
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching progress: {str(e)}")
        return jsonify({'error': 'Failed to fetch progress'}), 500

def award_achievement(user_id, achievement_id):
    """Award an achievement to a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user already has this achievement
        cursor.execute('''
        SELECT id FROM user_achievements
        WHERE user_id = ? AND achievement_id = ?
        ''', (user_id, achievement_id))
        
        if cursor.fetchone():
            conn.close()
            return False
        
        # Get achievement details
        cursor.execute('SELECT xp_reward FROM achievements WHERE id = ?', (achievement_id,))
        achievement = cursor.fetchone()
        
        if not achievement:
            conn.close()
            return False
        
        # Award achievement and XP
        cursor.execute('''
        INSERT INTO user_achievements (id, user_id, achievement_id)
        VALUES (?, ?, ?)
        ''', (str(uuid.uuid4()), user_id, achievement_id))
        
        # Update user's XP and level
        cursor.execute('''
        UPDATE users
        SET experience_points = experience_points + ?,
            level = (experience_points + ?) / 100 + 1
        WHERE id = ?
        ''', (achievement['xp_reward'], achievement['xp_reward'], user_id))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error awarding achievement: {str(e)}")
        return False

def check_achievements(user_id):
    """Check and award achievements based on user's progress"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user's progress
        cursor.execute('''
        SELECT COUNT(*) as sign_count,
               SUM(practice_count) as total_practice,
               AVG(accuracy_score) as avg_accuracy
        FROM learning_progress
        WHERE user_id = ?
        ''', (user_id,))
        
        progress = cursor.fetchone()
        
        # Get user's post count
        cursor.execute('SELECT COUNT(*) as post_count FROM posts WHERE user_id = ?', (user_id,))
        post_count = cursor.fetchone()['post_count']
        
        # Get user's like count
        cursor.execute('''
        SELECT COUNT(*) as like_count
        FROM likes l
        JOIN posts p ON l.post_id = p.id
        WHERE p.user_id = ?
        ''', (user_id,))
        
        like_count = cursor.fetchone()['like_count']
        
        # Check and award achievements
        if progress['sign_count'] >= 1:
            award_achievement(user_id, 'first_sign')
        
        if progress['sign_count'] >= 10:
            award_achievement(user_id, 'practice_master')
        
        if post_count >= 1:
            award_achievement(user_id, 'community_contributor')
        
        if like_count >= 5:
            award_achievement(user_id, 'helpful_member')
        
        if progress['avg_accuracy'] >= 0.9 and progress['sign_count'] >= 5:
            award_achievement(user_id, 'sign_expert')
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking achievements: {str(e)}")

@app.route('/api/learning-progress', methods=['GET', 'POST'])
def api_learning_progress():
    """Get or update learning progress"""
    if request.method == 'GET':
        try:
            username = request.args.get('username', '').strip()
            if not username:
                return jsonify({'error': 'Username is required'}), 400
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get user ID
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return jsonify({'error': 'User not found'}), 404
            
            cursor.execute('''
            SELECT * FROM learning_progress
            WHERE user_id = ?
            ORDER BY last_practiced DESC
            ''', (user['id'],))
            
            progress = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({'progress': progress})
            
        except Exception as e:
            logger.error(f"Error fetching learning progress: {str(e)}")
            return jsonify({'error': 'Failed to fetch learning progress'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            sign_word = data.get('sign_word', '').strip()
            accuracy = float(data.get('accuracy', 0))
            
            if not username or not sign_word:
                return jsonify({'error': 'Username and sign word are required'}), 400
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get or create user
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if not user:
                user_id = str(uuid.uuid4())
                cursor.execute('''
                INSERT INTO users (id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
                ''', (user_id, username, f"{username}@temp.com", "no_password"))
            else:
                user_id = user['id']
            
            # Update learning progress
            cursor.execute('''
            INSERT INTO learning_progress (id, user_id, sign_word, practice_count, accuracy_score)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(user_id, sign_word) DO UPDATE SET
                practice_count = practice_count + 1,
                accuracy_score = (accuracy_score * practice_count + ?) / (practice_count + 1),
                last_practiced = CURRENT_TIMESTAMP
            ''', (str(uuid.uuid4()), user_id, sign_word, accuracy, accuracy))
            
            # Check for achievements
            check_achievements(user_id)
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Learning progress updated successfully'})
            
        except Exception as e:
            logger.error(f"Error updating learning progress: {str(e)}")
            return jsonify({'error': 'Failed to update learning progress'}), 500

@app.route('/api/profile', methods=['GET', 'PUT'])
def api_profile():
    """Get or update user profile"""
    user_id = "demo_user"  # Replace with actual user session management
    
    if request.method == 'GET':
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return jsonify({'user': dict(user)})
            else:
                return jsonify({'error': 'User not found'}), 404
                
        except Exception as e:
            logger.error(f"Error fetching profile: {str(e)}")
            return jsonify({'error': 'Failed to fetch profile'}), 500
    
    elif request.method == 'PUT':
        try:
            data = request.get_json()
            bio = data.get('bio', '').strip()
            learning_level = data.get('learning_level', 'beginner')
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE users 
            SET bio = ?, learning_level = ?
            WHERE id = ?
            ''', (bio, learning_level, user_id))
            
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Profile updated successfully'})
            
        except Exception as e:
            logger.error(f"Error updating profile: {str(e)}")
            return jsonify({'error': 'Failed to update profile'}), 500

@app.route('/community_video/<filename>')
def serve_community_video(filename):
    """Serve community uploaded videos"""
    try:
        return send_file(
            os.path.join(app.config['COMMUNITY_UPLOADS'], filename),
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Error serving community video: {str(e)}")
        return jsonify({'error': 'Video not found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ASL gesture from uploaded video, translate to Kannada, and generate speech."""
    if not model or not dataset:
        return jsonify({"error": "Model not initialized"}), 500
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    try:
        # Save uploaded file temporarily
        temp_filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        video_file.save(temp_path)
        
        try:
            # Preprocess video
            frames = preprocess_video(temp_path, max_frames=16, img_size=224)
            frames = frames.unsqueeze(0).to(device)  # Shape: [1, 3, 16, 224, 224]
            
            # Make prediction
            with torch.no_grad():
                outputs = model(frames)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_label = dataset.label_encoder.inverse_transform([predicted.item()])[0]
                confidence_score = confidence.item()
            
            # Translate predicted sign to Kannada
            kannada_translation = translate_to_kannada(predicted_label)
            if kannada_translation.startswith("Error"):
                logger.error(kannada_translation)
                kannada_translation = predicted_label  # Fallback to original label if translation fails
            
            # Generate speech for the translated text
            audio_file = text_to_speech(kannada_translation)
            if audio_file.startswith("Error"):
                logger.error(audio_file)
                audio_file = None  # Set to None if TTS fails
            
            # Clean up temporary file
            os.remove(temp_path)
            
            logger.info(f"Prediction: {predicted_label} (confidence: {confidence_score:.4f}), Kannada: {kannada_translation}")
            
            response = {
                "prediction": predicted_label,
                "confidence": confidence_score,
                "kannada_translation": kannada_translation
            }
            if audio_file:
                response["audio_file"] = audio_file
            
            return jsonify(response)
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate sign language video from text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        input_text = data['text'].strip()
        if not input_text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.mp4"
        
        # Path to WLASL video dataset
        base_dir = "wlasl_data/videos"
        
        # Generate video
        success, result, missing_words = create_sign_video(
            input_text, 
            base_dir, 
            filename
        )
        
        if success:
            logger.info(f"Video generated successfully: {filename}")
            words_processed = len(input_text.split()) - len(missing_words or [])
            
            return jsonify({
                "message": "Video generated successfully",
                "filename": filename,
                "words_processed": words_processed,
                "missing_words": missing_words or []
            })
        else:
            logger.error(f"Video generation failed: {result}")
            return jsonify({"error": result}), 500
            
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Serve generated video or audio file."""
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(
            file_path,
            as_attachment=False,  # Allow playback in browser
            mimetype='audio/mp3' if safe_filename.endswith('.mp3') else 'video/mp4'
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "model_loaded": model is not None,
            "dataset_loaded": dataset is not None,
            "device": str(device) if device else None,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({"error": "Endpoint not found"}), 404

def cleanup_old_files():
    """Clean up old temporary files"""
    try:
        current_time = datetime.now()
        
        # Clean up upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if (current_time - file_time).total_seconds() > 3600:  # 1 hour
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")
        
        # Clean up output folder (keep files for 24 hours)
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if (current_time - file_time).total_seconds() > 86400:  # 24 hours
                    os.remove(file_path)
                    logger.info(f"Cleaned up old generated video: {filename}")
                    
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

@app.route('/gamified_video/<filename>')
def serve_gamified_video(filename):
    """Serve gamified learning videos"""
    try:
        return send_file(
            os.path.join('gamified_learning_videos', filename),
            mimetype='video/mp4'
        )
    except Exception as e:
        logger.error(f"Error serving gamified video: {str(e)}")
        return jsonify({'error': 'Video not found'}), 404

@app.route('/api/record-and-predict', methods=['POST'])
def record_and_predict():
    """Handle video recording and prediction for text-to-sign feature"""
    if not model or not dataset:
        return jsonify({"error": "Model not initialized"}), 500
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    try:
        # Save uploaded file temporarily
        temp_filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        video_file.save(temp_path)
        
        try:
            # Preprocess video
            frames = preprocess_video(temp_path, max_frames=16, img_size=224)
            frames = frames.unsqueeze(0).to(device)  # Shape: [1, 3, 16, 224, 224]
            
            # Make prediction
            with torch.no_grad():
                outputs = model(frames)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_label = dataset.label_encoder.inverse_transform([predicted.item()])[0]
                confidence_score = confidence.item()
            
            # Translate predicted sign to Kannada
            kannada_translation = translate_to_kannada(predicted_label)
            if kannada_translation.startswith("Error"):
                logger.error(kannada_translation)
                kannada_translation = predicted_label  # Fallback to original label if translation fails
            
            # Generate speech for both English and Kannada
            english_audio = text_to_speech(predicted_label)
            kannada_audio = text_to_speech(kannada_translation)
            
            if english_audio.startswith("Error"):
                logger.error(english_audio)
                english_audio = None
            if kannada_audio.startswith("Error"):
                logger.error(kannada_audio)
                kannada_audio = None
            
            # Clean up temporary file
            os.remove(temp_path)
            
            logger.info(f"Prediction: {predicted_label} (confidence: {confidence_score:.4f}), Kannada: {kannada_translation}")
            
            response = {
                "prediction": predicted_label,
                "confidence": confidence_score,
                "kannada_translation": kannada_translation
            }
            if english_audio:
                response["english_audio"] = english_audio
            if kannada_audio:
                response["kannada_audio"] = kannada_audio
            
            return jsonify(response)
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/record-and-generate', methods=['POST'])
def record_and_generate():
    """Handle video recording and generate sign language video"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Save uploaded file temporarily
        temp_filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        video_file.save(temp_path)
        
        try:
            # Preprocess video
            frames = preprocess_video(temp_path, max_frames=16, img_size=224)
            frames = frames.unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(frames)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_label = dataset.label_encoder.inverse_transform([predicted.item()])[0]
                confidence_score = confidence.item()
            
            # Generate sign language video for the predicted text
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.mp4"
            base_dir = "wlasl_data/videos"
            
            success, result, missing_words = create_sign_video(
                predicted_label,
                base_dir,
                filename
            )
            
            if success:
                logger.info(f"Video generated successfully: {filename}")
                words_processed = len(predicted_label.split()) - len(missing_words or [])
                
                # Generate audio for both English and Kannada
                english_audio = text_to_speech(predicted_label)
                kannada_translation = translate_to_kannada(predicted_label)
                kannada_audio = text_to_speech(kannada_translation)
                
                response = {
                    "message": "Video generated successfully",
                    "filename": filename,
                    "words_processed": words_processed,
                    "missing_words": missing_words or [],
                    "prediction": predicted_label,
                    "confidence": confidence_score,
                    "kannada_translation": kannada_translation
                }
                
                if english_audio:
                    response["english_audio"] = english_audio
                if kannada_audio:
                    response["kannada_audio"] = kannada_audio
                
                return jsonify(response)
            else:
                logger.error(f"Video generation failed: {result}")
                return jsonify({"error": result}), 500
                
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/camera-record', methods=['POST'])
def camera_record():
    """Handle camera recording and prediction"""
    if not model or not dataset:
        return jsonify({"error": "Model not initialized"}), 500
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    try:
        # Save uploaded file temporarily
        temp_filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        video_file.save(temp_path)
        
        try:
            # Preprocess video
            frames = preprocess_video(temp_path, max_frames=16, img_size=224)
            frames = frames.unsqueeze(0).to(device)  # Shape: [1, 3, 16, 224, 224]
            
            # Make prediction
            with torch.no_grad():
                outputs = model(frames)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_label = dataset.label_encoder.inverse_transform([predicted.item()])[0]
                confidence_score = confidence.item()
            
            # Translate predicted sign to Kannada
            kannada_translation = translate_to_kannada(predicted_label)
            if kannada_translation.startswith("Error"):
                logger.error(kannada_translation)
                kannada_translation = predicted_label  # Fallback to original label if translation fails
            
            # Generate speech for the translated text
            audio_file = text_to_speech(kannada_translation)
            if audio_file.startswith("Error"):
                logger.error(audio_file)
                audio_file = None  # Set to None if TTS fails
            
            # Clean up temporary file
            os.remove(temp_path)
            
            logger.info(f"Prediction: {predicted_label} (confidence: {confidence_score:.4f}), Kannada: {kannada_translation}")
            
            response = {
                "prediction": predicted_label,
                "confidence": confidence_score,
                "kannada_translation": kannada_translation
            }
            if audio_file:
                response["audio_file"] = audio_file
            
            return jsonify(response)
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Initialize model
    logger.info("Initializing ASL recognition model...")
    if initialize_model():
        logger.info("Model initialized successfully!")
    else:
        logger.warning("Model initialization failed. Some features may not work.")
    
    # Clean up old files on startup
    cleanup_old_files()
    
    # Start the Flask application
    logger.info("Starting ASL Recognition & Community Platform...")
    app.run(host='0.0.0.0', port=5000, debug=True)
