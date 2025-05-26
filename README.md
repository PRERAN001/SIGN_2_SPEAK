# 🧠✋ sign_2_speak

**sign_2_speak** is an AI-powered communication bridge designed for the dumb and deaf community. It enables **real-time sign language recognition**, converts it to **text**, and then to **speech** using [Dhwani.ai](https://dhwani.ai/). It also offers **reverse translation**—transforming text back into signs—enabling **two-way communication** between non-signers and signers. 

But we didn’t stop there—we added **gamified learning** for engagement and a **community feature** to empower users to share region-specific signs and expand their understanding of different signing dialects.

---

## 🚀 Features

- 🔤 **Sign-to-Text-to-Speech Conversion** using AI & Dhwani.ai
- ↩️ **Text-to-Sign Rendering** for two-way communication
- 🧠 **Trained with WLASL Dataset** (Wide-scale American Sign Language)
- 🎮 **Gamified Learning** for interactive sign language practice
- 🌐 **Community Platform** to share local/regional signs and interact
- 📊 **Model Accuracy:** 84% on validation dataset

---

## 🧩 Problems It Solves

- 🧏 Bridges the communication gap between dumb/deaf individuals and the rest of the world
- 💬 Enables real-time, two-way conversations using sign language and voice
- 🎓 Makes learning sign language engaging and interactive through games
- 🌍 Promotes knowledge of **regional sign variations** via community sharing
- 🔄 Helps both signers and non-signers understand and interact more inclusively

---

## 🧠 How It Works

The system is trained on the [WLASL dataset](https://www.kaggle.com/datasets/dxye/isolated-sign-language-dataset) and uses a custom-trained deep learning model (achieving **84% accuracy**) to recognize isolated signs. It translates those into English text, then leverages **Dhwani.ai** for high-quality voice synthesis.

The reverse flow uses a sign video renderer to play the appropriate sign for a given text phrase.

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sign_2_speak.git
cd sign_2_speak
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Prepare the dataset
Run this script to download, preprocess, and organize the WLASL dataset:

bash
Copy
Edit
python download_wlasl_custom_and_train_once.py
4. Train the model
bash
Copy
Edit
python train_only.py
5. Run predictions or test accuracy
bash
Copy
Edit
python actual_prediction.py
6. Launch the web application
bash
Copy
Edit
python app.py
🌟 Bonus Features
Gamified Learning:
Watch sign videos and select the correct meaning. A fun and educational quiz for the hearing/speech impaired.

Regional Sign Community:
Connect with others, share local sign dialects, and expand your communication toolkit.

🛠️ Tech Stack
Python (OpenCV, TensorFlow, Keras)

Flask for the web backend

MongoDB Compass for storing community data

HTML, CSS, JavaScript for frontend

Dhwani.ai API for voice synthesis

WLASL Dataset for model training

📸 Screenshots & Demo
(Add screenshots or demo link here when available)

🤝 Contributing
Contributions, feedback, and suggestions are welcome! Feel free to fork the repo and submit a PR.

📄 License
MIT License

✨ Acknowledgments
WLASL Dataset

Dhwani.ai for voice API

OpenAI for assistance in development
