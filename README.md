# 🌾 FarmAI — Smart Farming Intelligence System

An intelligent agricultural decision-support system powered by **Machine Learning**, **Deep Learning**, and **Google Gemini AI** to recommend crops, detect plant diseases, and diagnose cow diseases in real-time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat-square&logo=flask)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-1.5_Flash-purple?style=flat-square&logo=google)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## 📋 Table of Contents
- [Features](#-features)
- [Models & Capabilities](#-models--capabilities)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Technologies Used](#-technologies-used)
- [Training](#-training)
- [Performance Metrics](#-performance-metrics)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

### 🌱 **Crop Recommendation System**
- **Input Parameters**: Soil NPK levels, Temperature, Humidity, pH, Rainfall
- **Output**: Primary crop recommendation with confidence score + top 3 alternatives
- **Accuracy**: 99.32% on test dataset
- **Real-time Predictions**: Instant results using a trained Random Forest model

### 🔍 **Plant Disease Detection**
- **Image Upload**: Drag-and-drop or click-to-upload
- **Disease Classification**: 38 diseases across 14 plant types
- **Confidence Scoring**: Top 3 predictions with probabilities
- **Model**: Fine-tuned MobileNetV2 CNN (Transfer Learning)

### 🐄 **Cow Disease Detection** *(New!)*
- **Image-Based Diagnosis**: Upload a photo of the cow for instant diagnosis
- **3 Conditions Detected**: Foot-and-Mouth Disease, Lumpy Skin Disease, Healthy
- **Severity Levels**: Critical / High / None — with actionable descriptions
- **Model**: Custom-trained MobileNetV2 on a cow disease dataset

### 🤖 **Gemini AI Treatment Advisor** *(New!)*
- **Powered by** `gemini-1.5-flash` — Google's latest generative AI model
- **Plant Disease Advice**: Pesticide names, dosages, organic alternatives, cultural practices, safety tips
- **Cow Disease Advice**: Medicines, home care, prevention, vet visit urgency
- **Smart Fallback**: A comprehensive built-in knowledge base (30+ plant diseases, 3 cow conditions) kicks in if Gemini is unavailable — so advice is **always available**
- **Structured Output**: Returns clean JSON directly parsed into an interactive UI panel

### 🎨 **Modern UI/UX**
- Beautiful gradient design with dark/light responsive layout
- Tabbed navigation: Crop Recommendation | Plant Disease | Cow Disease
- Smooth animations, real-time loading spinners
- Dynamic AI Advice panels with urgency badges and structured tables
- Mobile-friendly interface (also available as a React Native app)

---

## 📊 Models & Capabilities

### 🌾 Crop Recommendation — 22 Crops
```
Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes,
Jute, Kidney Beans, Lentil, Maize, Mango, Moth Beans, Mung Bean,
Muskmelon, Orange, Papaya, Pigeon Peas, Pomegranate, Rice, Watermelon
```

### 🌿 Plant Disease Detection — 38 Diseases across 14 Plants
```
Apple (4):          Apple Scab, Black Rot, Cedar Apple Rust, Healthy
Blueberry (1):      Healthy
Bell Pepper (2):    Bacterial Spot, Healthy
Cherry (2):         Healthy, Powdery Mildew
Corn/Maize (4):     Cercospora Leaf Spot, Common Rust, Healthy, Northern Leaf Blight
Grape (4):          Black Rot, Esca (Black Measles), Healthy, Leaf Blight
Orange (1):         Haunglongbing (Citrus Greening)
Peach (2):          Bacterial Spot, Healthy
Potato (3):         Early Blight, Healthy, Late Blight
Raspberry (1):      Healthy
Soybean (1):        Healthy
Squash (1):         Powdery Mildew
Strawberry (2):     Healthy, Leaf Scorch
Tomato (10):        Bacterial Spot, Early Blight, Healthy, Late Blight,
                    Leaf Mold, Septoria Leaf Spot, Spider Mites,
                    Target Spot, Yellow Leaf Curl Virus, Mosaic Virus
```

### 🐄 Cow Disease Detection — 3 Conditions
```
✅ Healthy            — No treatment required
⚠️  Lumpy Skin Disease — High severity, reportable
🦷 Foot-and-Mouth    — Critical, notifiable disease
```

---

## 📁 Project Structure

```
farm-ml/
├── app.py                          # Flask backend (all routes + Gemini integration)
├── farm.ipynb                      # EDA & crop model training notebook
├── train_disease_model.ipynb       # Plant disease CNN training notebook
├── train_cow_model.ipynb           # Cow disease CNN training notebook
├── train_model.py                  # Crop recommendation model training script
├── fix_model.py                    # Model repair/conversion utility
├── requirements.txt                # Python dependencies
├── Crop_recommendation.csv         # Dataset for crop recommendation model
├── .gitignore                      # Git ignore rules
│
├── models/
│   ├── crop_model.pkl              # Trained RandomForest crop model
│   ├── scaler.pkl                  # Feature scaler for crop model
│   ├── plant_disease_model.h5      # Trained MobileNetV2 plant disease model
│   ├── disease_classes.pkl         # Plant disease class mappings
│   ├── best_cow_model.h5           # Trained cow disease detection model
│   └── cow_model_info.json         # Cow model class index metadata
│
├── static/
│   └── style.css                   # Application stylesheet
│
├── templates/
│   └── index.html                  # Main web interface (3 tabs + AI panels)
│
├── FarmAI/                         # React Native mobile application
│   ├── App.js
│   ├── package.json
│   └── assets/
│
└── README.md                       # This file
```

---

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4 GB+ RAM recommended
- 5 GB+ free disk space (for models and datasets)
- A **Google Gemini API Key** (free at [aistudio.google.com](https://aistudio.google.com))

### Step 1: Clone the Repository
```bash
git clone https://github.com/GaganThind-byte/farm_ml.git
cd farm_ml
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Environment Variables
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your_gemini_api_key_here"

# Linux/Mac
export GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at: **http://127.0.0.1:5000**

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Optional* | Google Gemini API key for AI treatment advice |

> *If not set, the app automatically falls back to its built-in knowledge base — no features will break.

Get your free Gemini API key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## 💻 Usage

### 🌱 Crop Recommendation
1. Go to the **"Crop Recommendation"** tab
2. Enter soil and environmental parameters:
   - Nitrogen (N): 0–140 ppm
   - Phosphorus (P): 5–145 ppm
   - Potassium (K): 5–205 ppm
   - Temperature: 8.8–43.7 °C
   - Humidity: 14–99.5 %
   - pH Level: 3.5–9.9
   - Rainfall: 20–298 mm
3. Click **"Get Crop Recommendation"**
4. View primary crop + top 3 alternatives with confidence scores

### 🌿 Plant Disease Detection
1. Go to the **"Disease Detection"** tab
2. Upload a clear photo of the plant leaf (PNG/JPG/JPEG, max 10 MB)
3. Click **"Analyze Plant Disease"**
4. View the detected disease, confidence, and top 3 predictions
5. Click **"Get AI Treatment Advice"** to receive a Gemini-powered treatment plan with:
   - Pesticide recommendations with dosage & frequency
   - Organic alternatives
   - Cultural practices
   - Safety precautions
   - Urgency level badge

### 🐄 Cow Disease Detection
1. Go to the **"Cow Disease"** tab
2. Upload a photo of the cow
3. Click **"Analyze Cow"**
4. View the detected condition (Healthy / Lumpy / Foot-and-Mouth) with severity
5. Click **"Get AI Vet Advice"** to receive a Gemini-powered veterinary plan with:
   - Medicines, dosage, and administration route
   - Home care steps
   - Prevention tips
   - Vet visit urgency

---

## 🔌 API Endpoints

### `POST /predict` — Crop Recommendation
```json
// Request
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 20.87, "humidity": 82,
  "ph": 6.50, "rainfall": 202.93
}

// Response
{
  "success": true,
  "primary_crop": "rice",
  "confidence": "99.00%",
  "emoji": "🍚",
  "description": "Requires high humidity and rainfall",
  "top_recommendations": [...]
}
```

### `POST /predict-disease` — Plant Disease Detection
```bash
curl -X POST -F "image=@leaf.jpg" http://127.0.0.1:5000/predict-disease
```
```json
// Response
{
  "success": true,
  "predicted_disease": "Tomato___Late_blight",
  "confidence": "94.12%",
  "top_3": [
    {"disease": "Tomato___Late_blight", "confidence": "94.12%"},
    ...
  ]
}
```

### `POST /predict-cow-disease` — Cow Disease Detection
```bash
curl -X POST -F "image=@cow.jpg" http://127.0.0.1:5000/predict-cow-disease
```
```json
// Response
{
  "success": true,
  "predicted_class": "lumpy",
  "confidence": "88.45%",
  "emoji": "⚠️",
  "severity": "Medium",
  "description": "Lumpy Skin Disease caused by a virus. Requires veterinary attention.",
  "top_3": [...]
}
```

### `POST /api/treatment-advice` — Gemini AI Advice
```json
// Request (plant)
{ "type": "plant", "disease": "Tomato___Late_blight" }

// Request (cow)
{ "type": "cow", "disease": "lumpy" }

// Response
{
  "success": true,
  "source": "gemini",   // or "fallback"
  "advice": {
    "urgency": "High",
    "pesticides": [{"name": "Metalaxyl+Mancozeb", "dose": "2g/L", "frequency": "Every 7 days"}],
    "organic_alternatives": ["Copper hydroxide spray", "Neem oil"],
    "cultural_practices": ["Remove infected leaves", "Avoid overhead irrigation"],
    "safety_precautions": ["Wear gloves and mask"],
    "note": "Apply in the morning to maximise absorption."
  }
}
```

### `GET /api/crop-info` — All Crop Information
```json
{
  "rice": {"emoji": "🍚", "description": "Requires high humidity and rainfall"},
  ...
}
```

---

## 🏗 Technologies Used

### Backend
| Technology | Purpose |
|---|---|
| **Flask 2.0+** | Web framework & REST API |
| **TensorFlow / Keras 2.13+** | Deep learning inference |
| **scikit-learn 1.8+** | Crop recommendation (RandomForest) |
| **Google Generative AI SDK** | Gemini 1.5 Flash integration |
| **Flask-CORS** | Cross-origin support for mobile clients |
| **NumPy / Pandas** | Numerical computing & data handling |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5** | Semantic structure |
| **CSS3** | Gradients, animations, responsive design |
| **Vanilla JavaScript** | Fetch API, dynamic UI updates |
| **Bootstrap Icons** | UI iconography |

### AI / ML Models
| Model | Task |
|---|---|
| **RandomForestClassifier** | Crop recommendation |
| **MobileNetV2 (Transfer Learning)** | Plant disease detection |
| **Custom CNN (MobileNetV2)** | Cow disease detection |
| **Gemini 1.5 Flash** | AI treatment advice generation |

---

## 🎓 Training

### Crop Recommendation Model
```bash
python train_model.py
# or explore with:
jupyter notebook farm.ipynb
```
- **Algorithm**: Random Forest Classifier (100 trees)
- **Features**: 7 (N, P, K, Temperature, Humidity, pH, Rainfall)
- **Classes**: 22 crops | **Accuracy**: 99.32%
- **Dataset**: Crop_recommendation.csv (2,200 samples)

### Plant Disease Detection Model
```bash
jupyter notebook train_disease_model.ipynb
```
- **Architecture**: MobileNetV2 + Custom Dense Layers (Transfer Learning)
- **Image Size**: 224×224 | **Batch Size**: 32 | **Epochs**: 15 + 15 (two-phase)
- **Data Augmentation**: Rotation ±40°, Brightness, Zoom ±30%, Horizontal/Vertical Flip
- **Classes**: 38 diseases | **Dataset**: PlantVillage (26,209 images)

### Cow Disease Detection Model
```bash
jupyter notebook train_cow_model.ipynb
```
- **Architecture**: MobileNetV2 (fine-tuned) + Custom Dense Layers
- **Image Size**: 224×224 | **Classes**: 3 (healthy, lumpy, foot-and-mouth)
- **Dataset**: Custom cattle disease image dataset

---

## 📈 Performance Metrics

| Model | Accuracy | Prediction Time |
|---|---|---|
| Crop Recommendation (RandomForest) | **99.32%** | < 50 ms |
| Plant Disease Detection (MobileNetV2) | ~92–95% | ~200–300 ms |
| Cow Disease Detection (MobileNetV2) | ~85–90% | ~200–300 ms |
| Gemini AI Advice | Generative | ~1–3 s |

---

## 🚀 Deployment

### Local Development
```bash
python app.py
# Runs on http://127.0.0.1:5000
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Cloud Platforms
- **Google Cloud Run** — Recommended (native Gemini integration)
- **Railway / Render** — Easy Python deployments
- **Heroku** — Push-to-deploy
- **AWS EC2 / GCP Compute Engine** — Full control

---

## 🎯 Future Improvements

- [x] ~~Plant disease detection~~
- [x] ~~Cow disease detection~~
- [x] ~~Gemini AI treatment advice~~
- [x] ~~React Native mobile app~~
- [ ] IoT sensor integration for real-time soil monitoring
- [ ] Weather forecast API integration
- [ ] Pest detection model
- [ ] Historical scan tracking & analytics dashboard
- [ ] Multi-language support (Hindi, Punjabi, Telugu, etc.)
- [ ] Automated SMS/WhatsApp alerts for farmers
- [ ] Farmer community forum

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Gagandeep Singh**
- GitHub: [@GaganThind-byte](https://github.com/GaganThind-byte)

---

## 📚 References & Acknowledgments

- [PlantVillage Dataset](https://plantvillage.psu.edu/)
- [MobileNetV2 Paper — Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
- [Google Gemini AI](https://deepmind.google/technologies/gemini/)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Made with ❤️ for farmers & agricultural technology enthusiasts**

⭐ If you found this helpful, please give it a star on GitHub!
