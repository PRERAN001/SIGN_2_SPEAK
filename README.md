# Sign-2-Speak 🤟

An AI-powered American Sign Language (ASL) recognition and learning platform that bridges communication gaps through advanced computer vision and community-driven learning with the help of Dwani.ai

## 🌟 Features

### 🎯 Core Functionality
- **Real-time ASL Recognition**: Upload videos or use your camera to recognize ASL gestures
- **Text-to-Sign Generation**: Convert text input into sign language videos
- **Multi-language Support**: Automatic translation to Kannada with text-to-speech
- **High Accuracy**: Powered by ResNet3D deep learning model trained on WLASL dataset

### 🎮 Gamified Learning
- **Interactive Quizzes**: Test your ASL knowledge with video-based questions
- **Difficulty Levels**: Beginner to advanced learning paths

### 👥 Community Features
- **Social Learning**: Share videos and learn from the community
- **Discussion Forums**: Post questions, tips, and experiences
- **Like & Comment System**: Engage with other learners
- **User Profiles**: Track achievements and learning progress

### 📱 Modern Interface
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Dark Mode Support**: Comfortable viewing in any lighting
- **Intuitive UI**: Clean, accessible design with smooth animations
- **Real-time Camera**: Browser-based video recording and processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- Webcam (for real-time recognition)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PRERAN001/SIGN_2_SPEAK.git
   cd SIGN_2_SPEAK
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained model**
   ```bash
   # Download best_resnet3d_wlasl.pth and place it in the root directory
   # Contact the repository maintainer for model access
   ```

4. **Set up the WLASL dataset**
   ```bash
   # Create wlasl_data directory structure:
   # wlasl_data/
   # ├── videos/
   # │   ├── word1/
   # │   ├── word2/
   # │   └── ...
   # └── download_info.json
   ```

5. **Configure environment variables**
   ```bash
   export DWANI_API_KEY="your_dwani_api_key"
   export DWANI_API_BASE_URL="https://dwani-dwani-api.hf.space"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the platform**
   Open your browser and navigate to `http://localhost:5000`

## 🏗️ Architecture

### Backend Components
- **Flask Web Server**: RESTful API endpoints and web interface
- **PyTorch Model**: ResNet3D for video-based ASL recognition
- **SQLite Database**: User data, community posts, and learning progress
- **Video Processing**: OpenCV for frame extraction and preprocessing
- **Translation Service**: Google Translator API integration
- **Text-to-Speech**: Dhwani API for Kannada audio generation

### Frontend Features
- **Responsive Web Interface**: HTML5, CSS3, and vanilla JavaScript
- **Camera Integration**: WebRTC for real-time video capture
- **File Upload**: Drag-and-drop video upload functionality
- **Real-time Updates**: Dynamic content loading and status updates

### Database Schema
- **Users**: Profile information, learning levels, and achievements
- **Posts**: Community-generated content and videos
- **Learning Progress**: Individual sign practice and accuracy tracking
- **Achievements**: Gamification system with badges and rewards

## 📊 API Endpoints

### Recognition & Generation
- `POST /predict` - Recognize ASL from uploaded video
- `POST /generate` - Generate sign language video from text
- `POST /camera-record` - Process camera-recorded videos

### Community Features
- `GET/POST /api/posts` - Retrieve or create community posts
- `POST /api/posts/<id>/like` - Toggle likes on posts
- `GET/POST /api/posts/<id>/comments` - Manage post comments

### Learning System
- `GET /api/quiz-questions` - Fetch interactive quiz questions
- `POST /api/submit-quiz` - Submit quiz answers and update progress
- `GET /api/learning-videos` - Retrieve learning video content
- `GET/POST /api/learning-progress` - Track individual learning progress

### User Management
- `POST /api/register` - Create new user account
- `POST /api/login` - User authentication
- `GET/PUT /api/profile` - Manage user profiles
- `GET /api/achievements` - Retrieve user achievements

## 🎯 Usage Examples

### Video Recognition
```python
# Upload a video file for ASL recognition
curl -X POST -F "video=@sign_video.mp4" http://localhost:5000/predict
```

### Text-to-Sign Generation
```python
# Generate sign language video from text
curl -X POST -H "Content-Type: application/json" \\
     -d '{"text": "hello world"}' \\
     http://localhost:5000/generate
```

### Community Interaction
```python
# Create a new community post
curl -X POST -F "username=learner123" \\
     -F "title=My First Sign" \\
     -F "content=Learning ASL basics" \\
     -F "video=@my_sign.mp4" \\
     http://localhost:5000/api/posts
```

## 🔧 Configuration

### Model Configuration
- **Input Resolution**: 224x224 pixels
- **Frame Count**: 16 frames per video
- **Model Architecture**: ResNet3D-18
- **Dataset**: WLASL (Word-Level American Sign Language)

### Performance Optimization
- **GPU Acceleration**: CUDA support for faster inference
- **File Cleanup**: Automatic removal of temporary files
- **Caching**: Translation results cached for efficiency
- **Compression**: Video compression for storage optimization

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### Areas for Contribution
- **Model Improvements**: Enhance recognition accuracy
- **New Languages**: Add support for additional languages
- **UI/UX**: Improve user interface and experience
- **Documentation**: Expand guides and tutorials
- **Testing**: Add comprehensive test coverage

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Include error handling and logging

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WLASL Dataset**: Word-Level American Sign Language dataset
- **PyTorch Team**: Deep learning framework
- **OpenCV Community**: Computer vision library
- **Flask Framework**: Web application framework
- **Dhwani API**: Text-to-speech service for Kannada

## 📞 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Community**: Join our discussions in the community forum
- **Email**: Contact the maintainers at support@sign2speak.com

## 🔮 Roadmap

### Upcoming Features
- [ ] Mobile app development (iOS/Android)
- [ ] Real-time video streaming recognition
- [ ] Advanced gesture detection and tracking
- [ ] Multi-user video chat with ASL translation
- [ ] Offline mode for core functionality
- [ ] Integration with popular video conferencing platforms

### Long-term Goals
- [ ] Support for additional sign languages (BSL, JSL, etc.)
- [ ] AI-powered personalized learning paths
- [ ] Augmented reality (AR) sign language overlay
- [ ] Professional certification and assessment tools
- [ ] Educational institution partnerships

---

**Made with ❤️ for the deaf and hard-of-hearing community**

*Empowering communication through technology and community learning*
```

