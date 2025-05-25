import os
import cv2
import logging
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sign_video(input_text, base_dir, output_filename="output_custom.mp4", 
                     default_resolution=(224, 224), placeholder_color=(0, 0, 0)):
    """
    Create a combined sign language video from input text by concatenating videos for each word.
    
    Args:
        input_text (str): Space-separated words to convert to sign language video.
        base_dir (str or Path): Base directory containing word subfolders with .mp4 videos.
        output_filename (str): Name of the output video file.
        default_resolution (tuple): (width, height) for output video and resizing.
        placeholder_color (tuple): RGB color for placeholder frames if video is missing.
    
    Returns:
        bool: True if video was created successfully, False otherwise.
    """
    temp_dir = None
    try:
        # Convert paths to Path objects for cross-platform compatibility
        base_dir = Path(base_dir)
        output_dir = base_dir.parent / "combined_output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_filename

        # Create temporary directory for intermediate files
        temp_dir = Path(tempfile.mkdtemp())

        # Validate base directory
        if not base_dir.exists():
            logger.error(f"Base directory does not exist: {base_dir}")
            return False

        # Split input text into words
        words = input_text.lower().strip().split()
        if not words:
            logger.error("No valid words provided in input text")
            return False

        # Collect video paths
        video_paths = []
        for word in words:
            word_folder = base_dir / word
            if word_folder.exists():
                mp4_files = list(word_folder.glob("*.mp4"))
                if mp4_files:
                    video_paths.append((word, mp4_files[0]))
                    logger.info(f"Using video for '{word}': {mp4_files[0]}")
                else:
                    logger.warning(f"No .mp4 files found in folder: {word}")
                    video_paths.append((word, None))  # Placeholder for missing video
            else:
                logger.warning(f"Folder not found for word: {word}")
                video_paths.append((word, None))  # Placeholder for missing video

        # Initialize video writer parameters
        width, height = default_resolution
        fps = 20.0  # Default FPS, updated if valid video is found
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        frames_written = 0

        # Process videos
        for word, video_path in video_paths:
            if video_path is None:
                # Generate placeholder frames for missing video
                logger.info(f"Generating 1-second placeholder for '{word}'")
                placeholder_frame = np.full((height, width, 3), placeholder_color, dtype=np.uint8)
                if out is None:
                    temp_output = temp_dir / "temp_output.mp4"
                    out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
                    if not out.isOpened():
                        logger.error(f"Failed to initialize video writer")
                        return False
                for _ in range(int(fps)):  # 1-second placeholder at default FPS
                    out.write(placeholder_frame)
                    frames_written += 1
                continue

            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video for '{word}': {video_path}")
                continue

            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            if out is None:
                fps = video_fps  # Use first valid video's FPS
                temp_output = temp_dir / "temp_output.mp4"
                out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
                if not out.isOpened():
                    logger.error(f"Failed to initialize video writer")
                    cap.release()
                    return False

            # Read and write frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize frame to match output resolution
                frame = cv2.resize(frame, default_resolution, interpolation=cv2.INTER_AREA)
                out.write(frame)
                frames_written += 1

            cap.release()
            logger.info(f"Processed {word} with {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")

        # Finalize and clean up
        if out is not None:
            out.release()
            if frames_written > 0:
                # Move temporary file to final location
                shutil.move(str(temp_output), str(output_path))
                logger.info(f"Final video saved at: {output_path} with {frames_written} frames")
                return True
            else:
                logger.warning("No frames written. Check video files.")
                if temp_output.exists():
                    temp_output.unlink()
                return False
        else:
            logger.warning("No valid videos processed. Output not created.")
            return False

    except Exception as e:
        logger.error(f"Error in create_sign_video: {str(e)}")
        return False
    finally:
        # Clean up temporary directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Example usage
    test_text = "hello world"
    test_base_dir = Path("wlasl_data/videos")
    if test_base_dir.exists():
        create_sign_video(test_text, test_base_dir)
    else:
        logger.error(f"Test directory not found: {test_base_dir}")