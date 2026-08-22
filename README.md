# Skin-Disease-Detection-using-ML
# 🏥 Disease Prediction & AI Chatbot

An all-in-one Jupyter notebook application that combines a pre-trained deep learning model, FastAPI backend, and an interactive web UI for skin disease classification with AI-powered analysis and chatbot support.

**No React. No npm. No Docker. Everything in one notebook.**

---

## ✨ Features

- **🎯 Image-Based Disease Classification** — Upload a skin image and get instant predictions using a fine-tuned ResNet model
- **🤖 AI-Powered Analysis** — Get detailed AI-generated insights using Google Generative AI (Gemini)
- **💬 Interactive Chatbot** — Ask health-related questions and receive informative responses
- **🎨 Beautiful Web UI** — Modern, responsive interface served directly by FastAPI (no frontend build needed)
- **⚡ Fast Inference** — Uses PyTorch and GPU acceleration for real-time predictions
- **📊 Confidence Scores** — See model confidence percentages for each prediction
- **🔒 Privacy-Focused** — Run locally with no data sent to external servers (except Google API for AI features)

---

## 📋 Requirements

### System Requirements
- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**
- **GPU (recommended)** — CUDA 11.8+ for faster inference; CPU mode supported but slower

### Python Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pillow==10.1.0
torch==2.1.0
torchvision==0.16.0
numpy==1.24.0
google-generativeai==0.3.0
python-dotenv==1.0.0
```

All installed automatically when you run the first notebook cell.

### Pre-trained Model
- **File:** `fine_tuned_skin_disease_model_unfrozen.pth`
- **Size:** Typically 40–100 MB
- **Architecture:** ResNet18
- **Classes:** 8 skin diseases (see Configuration)

### API Keys
- **Google Generative AI Key** (optional but required for AI analysis & chatbot)
  - Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)
  - Free tier available

---

## 🚀 Quick Start

### 1. Prepare Your Files

Ensure you have:
- The Jupyter notebook file (`disease_prediction_app.ipynb`)
- Your trained model file (`fine_tuned_skin_disease_model_unfrozen.pth`) in the same directory
- A `.env` file with your Google API key (see below)

### 2. Create a `.env` File

In the same directory as the notebook, create a `.env` file:

```bash
# .env
GOOGLE_API_KEY=your_actual_api_key_here
```

**⚠️ Never commit this file to Git!** Add `.env` to your `.gitignore`.

### 3. Open and Run the Notebook

1. Open `disease_prediction_app.ipynb` in Jupyter
2. Go to **Cell → Run All** to execute every cell in order
3. Or run cells sequentially and wait for the server to start

### 4. Access the Web UI

Once the server is running, open your browser and visit:

```
http://localhost:8000
```

---

## ⚙️ Configuration

Edit the **Configuration** cell (Cell 3) before running the notebook:

```python
# Your Google API key (reads from .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Path to your trained model
MODEL_PATH = "fine_tuned_skin_disease_model_unfrozen.pth"

# Model architecture (detected from .pth)
MODEL_ARCHITECTURE = "resnet18"  # Options: resnet18, resnet34, resnet50, vgg16, efficientnet_b0

# Number of output classes
NUM_CLASSES = 8

# Class names in training order (REQUIRED)
CLASS_NAMES = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles",
]

# Image preprocessing
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Server settings
SERVER_PORT = 8000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

### ⚠️ Important: Class Names

The `.pth` file **does not store class names**. You must provide them manually in `CLASS_NAMES` in the **exact order** they were used during training.

**How to find the correct order:**
1. Check your training script/notebook for `train_dataset.classes`
2. If you used `torchvision.datasets.ImageFolder`, the order is **alphabetical by folder name**
3. Look at the order of folders in your training dataset directory

---

## 📱 How to Use

### 1. Disease Prediction Tab

1. **Upload an Image**
   - Click the dropzone or drag & drop a JPG/PNG/GIF
   - Max 10MB, min reasonable resolution (e.g., 224×224)

2. **Click "🚀 Predict Disease"**
   - Model inference (~1–3 seconds depending on hardware)
   - Result displays immediately with confidence score

3. **View AI Analysis**
   - AI-powered explanation with:
     - Possible condition description
     - Common symptoms
     - When to seek medical attention
     - General prevention tips
   - Loads asynchronously; prediction result shows first

4. **Reset**
   - Clear the image and try another

### 2. AI Assistant Tab

1. **Ask Questions** in the chat box
2. **Get Responses** — AI answers general health questions
3. **Full Conversation Context** — AI remembers the conversation

**Example questions:**
- "What are common symptoms of ringworm?"
- "How is athlete's foot transmitted?"
- "What should I do if my symptoms worsen?"

---

## 🔌 API Endpoints

All endpoints are accessible at `http://localhost:8000`:

### GET `/` — Web UI
Returns the interactive HTML interface.

### GET `/health` — Health Check
Returns system status:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_configured": true,
  "vision_configured": true
}
```

### POST `/predict` — Image Prediction
**Request:**
```
Content-Type: multipart/form-data
file: <image_file>
```

**Response:**
```json
{
  "disease": "FU-ringworm",
  "confidence": 87.45,
  "class_index": 4,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### POST `/analysis` — AI Analysis
**Request:**
```
Content-Type: multipart/form-data
file: <image_file>
disease: "FU-ringworm"
confidence: 87.45
```

**Response:**
```json
{
  "ai_analysis": "Ringworm is a fungal infection... [detailed explanation]"
}
```

### POST `/chat` — Chatbot
**Request:**
```json
{
  "message": "What are symptoms of chickenpox?"
}
```

**Response:**
```json
{
  "response": "Chickenpox symptoms include...",
  "role": "assistant"
}
```

### Interactive API Docs
FastAPI auto-generates Swagger UI documentation at:
```
http://localhost:8000/docs
```

---

## 🛠️ Troubleshooting

### Model Not Loading

**Error:** `Model file not found: fine_tuned_skin_disease_model_unfrozen.pth`

**Solution:**
- Verify the file is in the same directory as the notebook
- Update `MODEL_PATH` if using a different filename
- Re-run Cell 4

### Architecture Mismatch

**Error:** `size mismatch` or `Missing key(s)`

**Solution:**
1. Check your training script to confirm the exact architecture
2. Copy the model-building code from your training script into `_build_model()`
3. Ensure `NUM_CLASSES` matches your training setup
4. Re-run Cell 4

### Google API Key Not Working

**Error:** `GOOGLE_API_KEY not found` or `Could not list models`

**Solution:**
1. Create a `.env` file in the notebook directory with:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
2. Verify the key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)
3. Ensure the key has access to Generative Language API
4. Re-run Cell 5

### AI Analysis Unavailable

**Error:** `AI analysis unavailable — set GOOGLE_API_KEY`

**Solution:**
- If you don't have a Google API key, the prediction will still work
- Disease classification runs locally; only AI analysis requires the API key

### Server Already Running on Port 8000

**Error:** `Address already in use`

**Solution:**
- Use a different port by changing `SERVER_PORT = 9000` in Cell 3
- Or kill the existing process:
  ```bash
  lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9
  ```

### How to Stop the Server

- **In Jupyter:** Kernel → Restart (or Interrupt)
- **In Terminal:** Press `Ctrl+C` twice
- The server runs as a daemon thread, so restarting the kernel fully stops it

---

## 📊 Model Details

### Architecture: ResNet18
- **Input:** 224×224 RGB images
- **Output:** 8 disease classes with softmax probabilities
- **Backbone:** ResNet18 (18-layer residual network)
- **Training:** Fine-tuned on skin disease images

### Preprocessing
- Resize to 224×224
- Normalize with ImageNet statistics (mean & std)
- RGB conversion

### Confidence Score
The confidence % represents the model's softmax probability for the predicted class.
- **High (>85%):** Model is quite certain
- **Medium (70–85%):** Model is moderately confident
- **Low (<70%):** Model is uncertain; consider alternative diagnoses

---

## ⚕️ Medical Disclaimer

**This application is for educational purposes only.**

- **Not a diagnosis tool** — The model's output is a computational prediction, not a medical diagnosis
- **Consult healthcare professionals** — Any skin concerns should be evaluated by a qualified dermatologist or physician
- **No treatment advice** — This app does not provide treatment recommendations
- **AI analysis is informational** — Generated explanations are educational content, not medical advice

**Always seek professional medical evaluation for diagnosis and treatment.**

---

## 🔐 Privacy & Data

- **Model runs locally** — Your images stay on your machine
- **No external storage** — Images are not saved or logged
- **Google API** — Only used for AI analysis/chatbot (if API key is provided)
- **No tracking** — No analytics or usage monitoring

---

## 📝 Project Structure

```
.
├── disease_prediction_app.ipynb     # Main notebook (all-in-one app)
├── fine_tuned_skin_disease_model_unfrozen.pth  # Pre-trained model
├── .env                             # Google API key (not in Git)
└── .gitignore                       # Excludes .env, __pycache__, etc.
```

---

## 🎓 Learning Resources

### Understanding ResNet
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet Documentation](https://pytorch.org/vision/stable/models.html)

### FastAPI
- [Official Documentation](https://fastapi.tiangolo.com/)

### Google Generative AI
- [Gemini API Docs](https://ai.google.dev/docs)
- [Python SDK](https://github.com/google/generative-ai-python)

---

## 🔄 Customizing for Your Use Case

### Change Model Architecture
Edit `_build_model()` in Cell 4 to match your training code:
```python
def _build_model(self, architecture, num_classes):
    if architecture == "your_custom_model":
        model = YourCustomModel(num_classes)
    # ... add more architectures
    return model
```

### Update Class Names
Modify `CLASS_NAMES` in Cell 3 to match your training labels.

### Change Server Port
Update `SERVER_PORT = 9000` in Cell 3.

### Modify UI
Edit the `HTML_PAGE` string in Cell 7 to customize colors, text, layout, etc.

---

## ⚡ Performance Tips

1. **Use GPU** — CUDA support is auto-detected; no configuration needed
2. **Batch processing** — Currently handles one image at a time; modify `/predict` for batches
3. **Cache model** — Model is loaded once and reused for all predictions
4. **Chat history** — Limited to last 20 messages to save memory

---

## 🐛 Reporting Issues

If you encounter problems:

1. **Check the notebook output** — Error messages are often self-explanatory
2. **Verify configuration** — Re-read the Configuration section
3. **Test components separately:**
   - Model loading: Run Cell 4 in isolation
   - API key: Test in Cell 5
   - Server startup: Check Cell 9 output
4. **Restart the kernel** — Often fixes environment issues

---

## 📄 License

[Add your license here, e.g., MIT, Apache 2.0, etc.]

---

## 🙌 Contributing

Contributions are welcome! To improve this project:

1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

---

## 📞 Support

For questions or issues:
- Check this README's Troubleshooting section
- Review the notebook comments
- Consult FastAPI and PyTorch documentation

---

## 🎉 Acknowledgments

- **PyTorch** & **TorchVision** for deep learning framework
- **FastAPI** for the web framework
- **Google Generative AI** for Gemini API
- Built with ❤️ for accessible healthcare technology

---

**Last Updated:** January 2024  
**Python:** 3.8+  
**Status:** Active Development
