import sys, io
# Fix Windows console encoding so Unicode / emoji in print() never crashes
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json
import os
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# ── Load environment variables from .env ──────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # reads .env file in project root

# ── Gemini AI setup (google-genai SDK) ────────────────────────────────
try:
    from google import genai as _genai_sdk
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
    if GEMINI_API_KEY:
        genai_client = _genai_sdk.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL  = 'gemini-flash-lite-latest'
        print(f'✅ Gemini AI ready (google-genai SDK)!')
    else:
        genai_client = None
        GEMINI_MODEL  = None
        print('⚠️  No GEMINI_API_KEY found in .env — using built-in knowledge base for advice.')
except Exception as _ge:
    genai_client = None
    GEMINI_MODEL  = None
    print(f'⚠️  Gemini setup error: {_ge}')




app = Flask(__name__)
CORS(app)  # Allow React Native / mobile clients on the local network

# Load the trained crop recommendation model and scaler 
with open('models/crop_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the trained plant disease detection model
disease_model = None
disease_classes = None

print("\n" + "="*50)
print("Loading Plant Disease Detection Model...")
print("="*50)

try:
    print("[1/2] Loading model architecture from plant_disease_model.h5...")
    disease_model = load_model('models/plant_disease_model.h5')
    print("     ✓ Model loaded successfully")
    
    print("[2/2] Loading disease classes...")
    with open('models/disease_classes.pkl', 'rb') as f:
        disease_classes = pickle.load(f)
    print(f"     ✓ Disease classes loaded ({len(disease_classes)} classes)")
    
    print("\n✅ Plant disease detection model is READY!")
    print("="*50 + "\n")
except Exception as e:
    import traceback
    print(f"\n❌ ERROR loading plant disease model!")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Details: {str(e)}")
    print("\n   Full Traceback:")
    traceback.print_exc()
    print("\n   Plant disease prediction will be UNAVAILABLE.")
    print("="*50 + "\n")

# ── Load the trained COW disease detection model ───────────────────────
cow_model = None
cow_classes = None

COW_CLASS_LABELS = {
    'foot-and-mouth': {'emoji': '🦷', 'severity': 'High', 'description': 'Highly contagious viral disease affecting hooves and mouth. Isolate immediately and contact a vet.'},
    'healthy':        {'emoji': '✅', 'severity': 'None', 'description': 'Cow appears healthy. Continue regular monitoring and care.'},
    'lumpy':          {'emoji': '⚠️',  'severity': 'Medium', 'description': 'Lumpy Skin Disease caused by a virus. Requires veterinary attention and vaccination.'}
}

print("\n" + "="*50)
print("Loading Cow Disease Detection Model...")
print("="*50)

try:
    cow_model_path = 'models/best_cow_model.h5'
    if not os.path.exists(cow_model_path):
        cow_model_path = 'models/cow_disease_model.h5'
    print(f"[1/2] Loading cow model from {cow_model_path}...")
    cow_model = load_model(cow_model_path)
    print("     ✓ Cow model loaded successfully")

    # Derive class list from saved JSON metadata if available
    cow_info_path = 'models/cow_model_info.json'
    cow_pkl_path  = 'models/cow_classes.pkl'
    if os.path.exists(cow_info_path):
        with open(cow_info_path, 'r') as f:
            cow_meta = json.load(f)
        # class_indices maps name->index; sort by index to get ordered list
        idx_map  = cow_meta.get('class_indices', {})
        cow_classes = [k for k, _ in sorted(idx_map.items(), key=lambda x: x[1])]
        print(f"     ✓ Classes from JSON: {cow_classes}")
    elif os.path.exists(cow_pkl_path):
        with open(cow_pkl_path, 'rb') as f:
            cow_classes = pickle.load(f)
        print(f"     ✓ Classes from PKL: {cow_classes}")
    else:
        cow_classes = sorted(COW_CLASS_LABELS.keys())
        print(f"     ℹ Using default class order: {cow_classes}")

    print(f"\n✅ Cow disease model is READY! ({len(cow_classes)} classes)")
    print("="*50 + "\n")
except Exception as e:
    import traceback
    print(f"\n❌ ERROR loading cow disease model!")
    print(f"   Error Type: {type(e).__name__}")
    print(f"   Details: {str(e)}")
    traceback.print_exc()
    print("\n   Cow disease prediction will be UNAVAILABLE.")
    print("="*50 + "\n")

# Feature names
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Crop information with emojis and descriptions
CROP_INFO = {
    'rice': {'emoji': '🍚', 'description': 'Requires high humidity and rainfall'},
    'maize': {'emoji': '🌽', 'description': 'Versatile crop, moderate conditions'},
    'chickpea': {'emoji': '🫘', 'description': 'Legume, good for soil nitrogen'},
    'kidneybeans': {'emoji': '🫘', 'description': 'Nutrient-rich legume'},
    'pigeonpeas': {'emoji': '🫘', 'description': 'Drought tolerant legume'},
    'mothbeans': {'emoji': '🫘', 'description': 'Drought resistant pulse crop'},
    'mungbean': {'emoji': '🫘', 'description': 'Quick-growing legume'},
    'blackgram': {'emoji': '⚫', 'description': 'Winter crop legume'},
    'lentil': {'emoji': '🍲', 'description': 'Cool season crop'},
    'pomegranate': {'emoji': '🍎', 'description': 'Tree fruit, drought tolerant'},
    'banana': {'emoji': '🍌', 'description': 'Tropical perennial crop'},
    'mango': {'emoji': '🥭', 'description': 'Tropical tree fruit'},
    'grapes': {'emoji': '🍇', 'description': 'Vineyard crop'},
    'watermelon': {'emoji': '🍉', 'description': 'Summer fruit crop'},
    'muskmelon': {'emoji': '🍈', 'description': 'Melon crop'},
    'apple': {'emoji': '🍎', 'description': 'Temperate tree fruit'},
    'orange': {'emoji': '🍊', 'description': 'Citrus fruit'},
    'papaya': {'emoji': '🧡', 'description': 'Tropical fruit'},
    'coconut': {'emoji': '🥥', 'description': 'Tropical tree'},
    'cotton': {'emoji': '⚪', 'description': 'Fiber crop'},
    'jute': {'emoji': '🌾', 'description': 'Fiber crop'},
    'coffee': {'emoji': '☕', 'description': 'Beverage crop'},
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        
        # Extract features in the correct order
        features = [
            float(data.get('N', 0)),
            float(data.get('P', 0)),
            float(data.get('K', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('ph', 0)),
            float(data.get('rainfall', 0)),
        ]
        
        # Convert to numpy array and scale
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = crop_model.predict(features_scaled)[0]
        probabilities = crop_model.predict_proba(features_scaled)[0]
        
        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [
            {
                'crop': crop_model.classes_[idx],
                'confidence': f"{probabilities[idx]*100:.2f}%",
                'emoji': CROP_INFO.get(crop_model.classes_[idx], {}).get('emoji', '🌱'),
                'description': CROP_INFO.get(crop_model.classes_[idx], {}).get('description', '')
            }
            for idx in top_indices
        ]
        
        return jsonify({
            'success': True,
            'primary_crop': prediction,
            'confidence': f"{max(probabilities)*100:.2f}%",
            'emoji': CROP_INFO.get(prediction, {}).get('emoji', '🌱'),
            'description': CROP_INFO.get(prediction, {}).get('description', ''),
            'top_recommendations': top_crops
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    """Predict plant disease from uploaded image"""
    if disease_model is None or disease_classes is None:
        return jsonify({
            'success': False,
            'error': 'Disease detection model not available'
        }), 503
    
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        # Convert FileStorage to BytesIO for load_img
        image_bytes = BytesIO(image_file.read())
        
        # Load and preprocess image
        img = load_img(image_bytes, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = disease_model.predict(img_batch, verbose=0)
        confidence = np.max(prediction)
        predicted_idx = np.argmax(prediction)
        predicted_disease = disease_classes[predicted_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[::-1][:3]
        top_3 = [
            {
                'disease': disease_classes[idx],
                'confidence': f"{prediction[0][idx]*100:.2f}%"
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            'success': True,
            'predicted_disease': predicted_disease,
            'confidence': f"{confidence*100:.2f}%",
            'top_3': top_3
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict-cow-disease', methods=['POST'])
def predict_cow_disease():
    """Predict cow disease from uploaded image"""
    if cow_model is None or cow_classes is None:
        return jsonify({
            'success': False,
            'error': 'Cow disease detection model not available'
        }), 503

    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400

        image_bytes = BytesIO(image_file.read())
        img        = load_img(image_bytes, target_size=(224, 224))
        img_array  = img_to_array(img) / 255.0
        img_batch  = np.expand_dims(img_array, axis=0)

        prediction   = cow_model.predict(img_batch, verbose=0)
        confidence   = float(np.max(prediction))
        predicted_idx   = int(np.argmax(prediction))
        predicted_class = cow_classes[predicted_idx]

        top3_indices = np.argsort(prediction[0])[::-1][:3]
        top3 = [
            {
                'disease':    cow_classes[idx],
                'confidence': f"{prediction[0][idx]*100:.2f}%",
                'emoji':      COW_CLASS_LABELS.get(cow_classes[idx], {}).get('emoji', '🐄')
            }
            for idx in top3_indices
        ]

        label_info = COW_CLASS_LABELS.get(predicted_class, {})

        return jsonify({
            'success':          True,
            'predicted_class':  predicted_class,
            'confidence':       f"{confidence*100:.2f}%",
            'emoji':            label_info.get('emoji', '🐄'),
            'severity':         label_info.get('severity', 'Unknown'),
            'description':      label_info.get('description', ''),
            'top_3':            top3
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ── Helpers ──────────────────────────────────────────────────────────
def parse_disease_name(raw: str) -> tuple[str, str]:
    """
    Convert model class names to human-readable form.
    'Tomato___Late_blight'  -> plant='Tomato', disease='Late Blight'
    'Apple___healthy'       -> plant='Apple',  disease='Healthy'
    'foot-and-mouth'        -> plant='Cow',    disease='Foot And Mouth'
    """
    if '___' in raw:
        parts   = raw.split('___', 1)
        plant   = parts[0].replace('_', ' ').strip()
        disease = parts[1].replace('_', ' ').strip().title()
    else:
        plant   = 'Cow'
        disease = raw.replace('-', ' ').replace('_', ' ').title()
    return plant, disease


def build_gemini_prompt(disease_type: str, raw_name: str) -> str:
    plant, disease = parse_disease_name(raw_name)
    readable = f"{plant} — {disease}" if disease_type == 'plant' else disease

    if disease_type == 'cow':
        return (
            f"Indian cattle vet. Cow diagnosed: {readable}. "
            'Reply ONLY with this JSON (no markdown): '
            '{"urgency":"Critical|High|Medium|Low|None","medicines":[{"name":"...","type":"...","dosage":"...","route":"..."}],'
            '"home_care":["step"],"prevention":["tip"],"vet_visit":"...","note":"..."}'
        )
    else:
        return (
            f"Indian crop protection expert. {plant} diagnosed with {disease}. "
            'Reply ONLY with this JSON (no markdown): '
            '{"urgency":"High|Medium|Low","pesticides":[{"name":"...","type":"...","dose":"...","frequency":"..."}],'
            '"organic_alternatives":["..."],"cultural_practices":["..."],"safety_precautions":["..."],"note":"..."}'
        )


# ── Comprehensive fallback knowledge base ─────────────────────────────
PLANT_FALLBACK = {
    # Apple
    'Apple___Apple_scab':               ('Captan 50WP or Mancozeb 75WP', 'Medium', 'Prune infected branches; improve air circulation; apply dormant lime-sulphur spray'),
    'Apple___Black_rot':                ('Copper-based fungicide or Thiophanate-methyl', 'High', 'Remove mummified fruit; prune cankers; maintain clean orchard floor'),
    'Apple___Cedar_apple_rust':         ('Myclobutanil (Eagle 20EW) or Propiconazole', 'Medium', 'Remove nearby juniper host plants; apply fungicide at bud break'),
    'Apple___healthy':                  (None, 'None', 'Continue standard maintenance and scouting'),
    # Blueberry
    'Blueberry___healthy':              (None, 'None', 'Maintain soil pH 4.5-5.5; mulch well'),
    # Cherry
    'Cherry_(including_sour)___Powdery_mildew': ('Sulfur-based fungicide or Myclobutanil', 'Medium', 'Improve air flow; avoid overhead irrigation; prune dense canopy'),
    'Cherry_(including_sour)___healthy': (None, 'None', 'Continue standard maintenance'),
    # Corn
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ('Azoxystrobin or Propiconazole', 'High', 'Rotate crops; use resistant hybrids; avoid late planting'),
    'Corn_(maize)___Common_rust_':      ('Propiconazole or Azoxystrobin foliar spray', 'Medium', 'Plant resistant hybrids; early planting; monitor fields regularly'),
    'Corn_(maize)___Northern_Leaf_Blight': ('Mancozeb or Azoxystrobin', 'Medium', 'Crop rotation; use resistant varieties; destroy infected residue'),
    'Corn_(maize)___healthy':           (None, 'None', 'Continue standard crop monitoring'),
    # Grape
    'Grape___Black_rot':                ('Mancozeb 75WP or Myclobutanil', 'High', 'Remove mummified berries; ensure good canopy airflow; spray at bud break'),
    'Grape___Esca_(Black_Measles)':     ('No effective chemical cure — pruning is key', 'High', 'Prune infected wood; seal cuts with wound paint; remove severely affected vines'),
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ('Copper oxychloride or Mancozeb', 'Medium', 'Remove fallen leaves; improve airflow; avoid wetting foliage'),
    'Grape___healthy':                  (None, 'None', 'Continue standard vine management'),
    # Orange
    'Orange___Haunglongbing_(Citrus_greening)': ('No cure — vector control (Imidacloprid for psyllid) + remove infected trees', 'Critical', 'Control Asian citrus psyllid; use disease-free nursery stock; remove infected trees immediately'),
    # Peach
    'Peach___Bacterial_spot':           ('Copper hydroxide (Kocide) or Oxytetracycline', 'Medium', 'Use resistant cultivars; avoid overhead irrigation; remove infected fruit/leaves'),
    'Peach___healthy':                  (None, 'None', 'Continue standard maintenance'),
    # Pepper
    'Pepper,_bell___Bacterial_spot':    ('Copper-based bactericide (Kocide) spray', 'Medium', 'Use certified disease-free seed; avoid overhead irrigation; crop rotation'),
    'Pepper,_bell___healthy':           (None, 'None', 'Continue standard management'),
    # Potato
    'Potato___Early_blight':            ('Mancozeb 75WP or Chlorothalonil', 'Medium', 'Crop rotation; remove volunteer plants; ensure balanced nitrogen'),
    'Potato___Late_blight':             ('Metalaxyl+Mancozeb (Ridomil Gold) or Copper oxychloride', 'High', 'Use certified seed; avoid waterlogging; destroy infected haulm before harvest'),
    'Potato___healthy':                 (None, 'None', 'Maintain scouting and standard management'),
    # Raspberry
    'Raspberry___healthy':              (None, 'None', 'Continue standard cane management'),
    # Soybean
    'Soybean___healthy':                (None, 'None', 'Continue standard management and rotation'),
    # Squash
    'Squash___Powdery_mildew':          ('Sulfur dust or Potassium bicarbonate spray', 'Medium', 'Plant resistant varieties; improve spacing and airflow; avoid excess nitrogen'),
    # Strawberry
    'Strawberry___Leaf_scorch':         ('Captan or Copper-based fungicide', 'Medium', 'Remove infected leaves; avoid overhead irrigation; mulch around plants'),
    'Strawberry___healthy':             (None, 'None', 'Continue standard management'),
    # Tomato
    'Tomato___Bacterial_spot':          ('Copper hydroxide spray + Mancozeb (copper fixative)', 'High', 'Use certified disease-free transplants; avoid working in wet conditions; crop rotation'),
    'Tomato___Early_blight':            ('Mancozeb 75WP or Azoxystrobin fungicide', 'Medium', 'Mulch soil; remove lower infected leaves; stake plants for airflow'),
    'Tomato___Late_blight':             ('Metalaxyl+Mancozeb (Ridomil Gold) or Chlorothalonil', 'High', 'Avoid overhead irrigation; plant resistant varieties; 3-year crop rotation'),
    'Tomato___Leaf_Mold':               ('Chlorothalonil or Copper-based fungicide', 'Medium', 'Increase ventilation in greenhouses; reduce humidity; remove infected leaves'),
    'Tomato___Septoria_leaf_spot':      ('Mancozeb or Chlorothalonil spray', 'Medium', 'Remove infected leaves; mulch; stake for airflow; rotate crops'),
    'Tomato___Spider_mites Two-spotted_spider_mite': ('Abamectin (Vertimec) or Bifenazate miticide', 'Medium', 'Use miticide rotation to avoid resistance; introduce predatory mites; remove dusty weeds'),
    'Tomato___Target_Spot':             ('Azoxystrobin or Boscalid fungicide', 'Medium', 'Crop rotation; prune lower leaves; stake for airflow'),
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ('No chemical cure — Imidacloprid to control whitefly vector', 'High', 'Use whitefly-resistant varieties; install yellow sticky traps; remove infected plants'),
    'Tomato___Tomato_mosaic_virus':     ('No chemical cure — remove infected plants immediately', 'High', 'Control aphid vectors; disinfect tools; use resistant seed varieties'),
    'Tomato___healthy':                 (None, 'None', 'Continue standard scouting and management'),
}

COW_FALLBACK = {
    'foot-and-mouth': ('Antiseptic mouth wash (dilute KMnO4), wound spray, Meloxicam anti-inflammatory, FMD polyvalent vaccine', 'Critical — notify state vet authority (notifiable disease)', 'Routine FMD vaccination every 6 months; strict biosecurity; isolate new animals 21 days'),
    'lumpy':          ('LSD Neethling-strain vaccine; Flunixin anti-inflammatory; Oxytetracycline antibiotic for secondary infections; iodine wound spray', 'High — reportable disease', 'Vaccinate all cattle; control biting insects with pour-on insecticide; restrict movement of infected animals'),
    'healthy':        ('No treatment required. Continue routine Albendazole deworming and vaccination schedule.', 'None', 'Balanced diet; clean water; annual FMD+BQ+HS vaccination; regular vet check-up'),
}


def make_plant_advice(raw_name):
    """Build a plant advice dict from the knowledge base entry."""
    entry = PLANT_FALLBACK.get(raw_name)
    _, disease = parse_disease_name(raw_name)

    if entry is None:
        # Not in KB at all — build from parsed name
        return {
            'urgency': 'Medium',
            'pesticides': [{'name': 'Mancozeb 75WP or Copper oxychloride', 'type': 'fungicide', 'dose': '2g per litre water', 'frequency': 'Every 7-10 days'}],
            'organic_alternatives': ['Neem oil spray (5ml/litre)', 'Trichoderma viride bio-fungicide'],
            'cultural_practices': ['Remove and destroy infected plant parts', 'Avoid overhead irrigation', 'Maintain proper plant spacing for airflow', 'Practice 2-3 year crop rotation'],
            'safety_precautions': ['Wear gloves and mask during spraying', 'Do not spray near water bodies or in strong wind'],
            'note': f'General advice for {disease}. Consult your local KVK for crop-specific dosing.'
        }

    treatment, urgency, prevention = entry
    if treatment is None:  # Healthy
        return {
            'urgency': 'None',
            'pesticides': [],
            'organic_alternatives': ['Neem cake soil treatment (preventive)', 'Trichoderma-enriched compost'],
            'cultural_practices': prevention.split('; '),
            'safety_precautions': ['Scout fields weekly for early signs of disease'],
            'note': 'Plant appears healthy. Focus on prevention and regular monitoring.'
        }

    return {
        'urgency': urgency,
        'pesticides': [{'name': treatment, 'type': 'fungicide/bactericide', 'dose': '2-3g or ml per litre water', 'frequency': 'Every 7-14 days as needed'}],
        'organic_alternatives': ['Neem oil spray (5ml/litre)', 'Bordeaux mixture (copper sulphate + lime)'],
        'cultural_practices': prevention.split('; '),
        'safety_precautions': ['Wear gloves and mask', 'Wash hands after handling treated plants', 'Do not spray near water bodies'],
        'note': 'Follow label dosage. Rotate fungicides to prevent resistance.'
    }


def make_cow_advice(raw_name):
    """Build a cow advice dict from the knowledge base entry."""
    entry = COW_FALLBACK.get(raw_name)
    if entry is None:
        _, disease = parse_disease_name(raw_name)
        return {
            'urgency': 'High',
            'medicines': [{'name': 'Consult veterinarian for specific prescription', 'type': 'other', 'dosage': 'As prescribed', 'route': 'As prescribed'}],
            'home_care': ['Isolate the affected animal immediately', 'Provide clean water and soft feed', 'Monitor temperature twice daily'],
            'prevention': ['Maintain routine vaccination schedule', 'Ensure clean housing and good ventilation', 'Practice biosecurity for new animals'],
            'vet_visit': 'Yes — contact a licensed veterinarian immediately',
            'note': f'Specific advice for {disease}. Always consult a qualified vet before administering medicines.'
        }

    treatment, urgency, prevention = entry
    return {
        'urgency': urgency,
        'medicines': [{'name': treatment, 'type': 'combined', 'dosage': 'As prescribed by vet', 'route': 'Various'}],
        'home_care': ['Isolate affected animal', 'Provide clean water and soft palatable feed', 'Keep wounds clean and dry'],
        'prevention': prevention.split('; '),
        'vet_visit': 'Yes — contact licensed veterinarian immediately',
        'note': 'Always consult a licensed veterinarian before administering medicines.'
    }


@app.route('/api/treatment-advice', methods=['POST'])
def treatment_advice():
    """Get AI-powered treatment suggestions for detected disease"""
    data         = request.json or {}
    disease_type = data.get('type', 'plant')   # 'plant' or 'cow'
    disease_name = data.get('disease', '').strip()

    if not disease_name:
        return jsonify({'success': False, 'error': 'Disease name is required'}), 400

    # ── Try Gemini first — always, for every disease ─────────────────
    if genai_client:
        try:
            prompt   = build_gemini_prompt(disease_type, disease_name)
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            text = response.text.strip()
            # Strip markdown code fences if Gemini adds them
            if '```' in text:
                text = text.split('```', 1)[-1]  # after first ```
                text = text.split('```')[0]       # before closing ```
                if text.startswith('json'):
                    text = text[4:].strip()
            advice = json.loads(text.strip())
            return jsonify({'success': True, 'source': 'gemini', 'advice': advice})
        except Exception as e:
            print(f"Gemini error ({disease_name}): {e} — using knowledge base")

    # ── Fallback: comprehensive built-in knowledge base ──────────────────
    if disease_type == 'plant':
        advice = make_plant_advice(disease_name)
    else:
        advice = make_cow_advice(disease_name)

    return jsonify({'success': True, 'source': 'fallback', 'advice': advice})


@app.route('/api/crop-info')
def crop_info():
    """API endpoint to get all crop information"""
    return jsonify(CROP_INFO)


# ── Language metadata ─────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    'en':  {'name': 'English',   'greeting': 'Hello! I am your FarmAI assistant. How can I help you today?'},
    'hi':  {'name': 'Hindi',     'greeting': 'नमस्ते! मैं आपका FarmAI सहायक हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?'},
    'pa':  {'name': 'Punjabi',   'greeting': 'ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ FarmAI ਸਹਾਇਕ ਹਾਂ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?'},
    'te':  {'name': 'Telugu',    'greeting': 'నమస్కారం! నేను మీ FarmAI సహాయకుడిని. ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?'},
    'ta':  {'name': 'Tamil',     'greeting': 'வணக்கம்! நான் உங்கள் FarmAI உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவலாம்?'},
    'mr':  {'name': 'Marathi',   'greeting': 'नमस्कार! मी तुमचा FarmAI सहाय्यक आहे. आज मी तुम्हाला कशी मदत करू शकतो?'},
    'kn':  {'name': 'Kannada',   'greeting': 'ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ FarmAI ಸಹಾಯಕ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?'},
    'gu':  {'name': 'Gujarati',  'greeting': 'નમસ્તે! હું તમારો FarmAI સહાયક છું. આજે હું તમારી કેવી રીતે મદદ કરી શકું?'},
    'bn':  {'name': 'Bengali',   'greeting': 'নমস্কার! আমি আপনার FarmAI সহকারী। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?'},
}

FARMING_SYSTEM_PROMPT = (
    "You are FarmAI, a concise Indian agricultural expert. "
    "Respond ONLY in {language_name}. "
    "Give short, practical answers about crops, plant diseases, cattle health, soil science, and Indian farming schemes. "
    "Use simple words a rural farmer can understand. "
    "Format with numbered steps when listing. "
    "If question is unrelated to farming/agriculture/cattle, say: 'I specialize in farming. Please ask about crops, diseases, or animal health.'"
)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Multilingual conversational farming assistant powered by Gemini"""
    data     = request.json or {}
    message  = data.get('message', '').strip()
    lang_code = data.get('language', 'en')
    history  = data.get('history', [])   # list of {role, text} dicts

    if not message:
        return jsonify({'success': False, 'error': 'Message is required'}), 400

    lang_info = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES['en'])
    lang_name = lang_info['name']

    # ── Try Gemini ───────────────────────────────────────────────────
    if genai_client:
        try:
            system_prompt = FARMING_SYSTEM_PROMPT.format(language_name=lang_name)

            # Build full conversation string with history
            conversation = system_prompt + "\n\n"
            for turn in history[-10:]:   # Keep last 10 turns to stay within context
                role_label = "Farmer" if turn.get('role') == 'user' else "FarmAI"
                conversation += f"{role_label}: {turn.get('text', '')}\n"
            conversation += f"Farmer: {message}\nFarmAI:"

            response = genai_client.models.generate_content(
                model=GEMINI_MODEL, contents=conversation
            )
            reply    = response.text.strip()

            return jsonify({
                'success': True,
                'reply':   reply,
                'language': lang_code,
                'source':   'gemini'
            })
        except Exception as e:
            print(f"Chat Gemini error: {e}")

    # ── Smart offline fallback (works without any API key) ───────────
    reply = _offline_farming_reply(message, lang_code)
    return jsonify({'success': True, 'reply': reply, 'language': lang_code, 'source': 'offline'})


# ── Keyword-based offline farming assistant ────────────────────────────
def _offline_farming_reply(message: str, lang: str) -> str:
    """Return helpful farming guidance based on keywords, in English.
    (Gemini is needed for non-English responses — get a free key at aistudio.google.com)"""
    msg = message.lower()

    # --- Disease ---
    if any(k in msg for k in ['blight', 'rust', 'scab', 'rot', 'wilt', 'mildew', 'disease', 'sick', 'infection', 'fungus', 'bimari', 'rog']):
        return ("**Plant Disease Help** 🌿\n\n"
                "1. Upload a photo of the affected leaf in the **Plant Disease** tab for AI diagnosis.\n"
                "2. Common treatment: Spray **Mancozeb 75WP** (2g per litre) or **Copper Oxychloride** every 7 days.\n"
                "3. Organic option: Mix **neem oil (5ml) + mild soap (1ml)** per litre of water and spray in the evening.\n"
                "4. Remove and burn severely infected leaves — do not compost them.\n"
                "5. Ensure good air circulation between plants.\n\n"
                "_Tip: Use the Plant Disease tab above for a specific diagnosis and medicine recommendation._")

    # --- Soil / fertilizer ---
    if any(k in msg for k in ['soil', 'fertilizer', 'npk', 'nitrogen', 'phosphorus', 'potassium', 'urea', 'manure', 'mitti', 'khad']):
        return ("**Soil & Fertilizer Guide** 🌱\n\n"
                "**Basic NPK for most crops:**\n"
                "- Nitrogen (N): Promotes leafy growth — use Urea (46-0-0), 100-120 kg/hectare\n"
                "- Phosphorus (P): Root strength — use DAP (18-46-0), 60-80 kg/hectare at sowing\n"
                "- Potassium (K): Disease resistance — use MOP, 40-60 kg/hectare\n\n"
                "**Organic options:** Farmyard manure (10 tons/ha), vermicompost, green manure crops\n\n"
                "**Soil pH:** Most crops need pH 6.0-7.0. Test with a soil kit.\n"
                "- Acidic soil: Add agricultural lime\n"
                "- Alkaline soil: Add gypsum or sulphur\n\n"
                "_Use the Crop Recommendation tab above to get personalized fertilizer advice for your soil conditions._")

    # --- Water / irrigation ---
    if any(k in msg for k in ['water', 'irrigation', 'drip', 'flood', 'drought', 'pani', 'sinchai']):
        return ("**Irrigation Guide** 💧\n\n"
                "**Crop water needs (approximate):**\n"
                "- Rice: 1200-2000 mm per season\n"
                "- Wheat: 400-500 mm per season\n"
                "- Cotton: 700-1200 mm per season\n"
                "- Maize: 500-800 mm per season\n\n"
                "**Best methods:**\n"
                "1. **Drip irrigation** — saves 40-60% water, reduces diseases\n"
                "2. **Sprinkler** — good for vegetables and wheat\n"
                "3. **Furrow** — good for row crops like sugarcane\n\n"
                "**Signs of water stress:** Wilting in the morning, curling leaves, pale colour\n"
                "**Signs of overwatering:** Yellow lower leaves, root rot, waterlogging")

    # --- Crop recommendation ---
    if any(k in msg for k in ['crop', 'grow', 'plant', 'sow', 'kharif', 'rabi', 'fasal', 'which crop', 'best crop']):
        return ("**Crop Selection Tips** 🌾\n\n"
                "**Kharif crops (June-November):** Rice, Maize, Cotton, Groundnut, Soybean, Jowar\n"
                "**Rabi crops (October-March):** Wheat, Barley, Mustard, Chickpea, Lentil\n"
                "**Zaid (March-June):** Cucumber, Watermelon, Muskmelon, Moong\n\n"
                "**Factors to consider:**\n"
                "- Soil type (sandy / loamy / clay)\n"
                "- Rainfall or irrigation availability\n"
                "- Market demand in your area\n"
                "- Previous crop (crop rotation avoids soil depletion)\n\n"
                "_For a personalized recommendation, use the **Crop Recommendation** tab — enter your soil NPK, pH, temperature and rainfall._")

    # --- Cow / cattle ---
    if any(k in msg for k in ['cow', 'cattle', 'buffalo', 'goat', 'animal', 'milk', 'gaay', 'pashu', 'vet', 'foot', 'lumpy']):
        return ("**Cattle Health Guide** 🐄\n\n"
                "**Common cattle diseases in India:**\n"
                "- **Foot & Mouth Disease (FMD):** Blisters on mouth/hooves — vaccinate every 6 months\n"
                "- **Lumpy Skin Disease:** Skin nodules — isolate immediately, contact vet\n"
                "- **Mastitis:** Swollen/painful udder — clean teats before milking, treat with antibiotics\n"
                "- **Bloat:** Swollen left side — walk the animal, drench with turpentine oil (30ml)\n\n"
                "**Prevention:**\n"
                "1. Vaccinate on schedule (FMD, HS, BQ)\n"
                "2. Deworm every 3-4 months\n"
                "3. Clean the shed daily\n"
                "4. Provide mineral mixture in feed\n\n"
                "_For specific disease diagnosis, upload a photo in the **Cow Disease** tab above._")

    # --- Weather / season ---
    if any(k in msg for k in ['weather', 'rain', 'temperature', 'frost', 'hail', 'season', 'mausam', 'barish']):
        return ("**Weather & Farming Tips** 🌤️\n\n"
                "- **Before sowing:** Check IMD (mausam.imd.gov.in) for 15-day forecast\n"
                "- **Frost protection:** Cover crops with cloth at night, irrigate before frost (water releases heat)\n"
                "- **Hail protection:** Use nets for high-value crops like vegetables and grapes\n"
                "- **Heavy rain:** Ensure drainage channels are clear to prevent waterlogging\n"
                "- **Drought:** Mulching (dry grass/straw on soil surface) reduces water evaporation by 30-40%\n\n"
                "**Free weather apps for farmers:**\n"
                "- Meghdoot app (government of India)\n"
                "- Kisan Suvidha app\n"
                "- IMD Agrimet")

    # --- Pest ---
    if any(k in msg for k in ['pest', 'insect', 'bug', 'locust', 'aphid', 'whitefly', 'mealy', 'keeda', 'kida']):
        return ("**Pest Management** 🐛\n\n"
                "**Common pests and solutions:**\n"
                "- **Aphids:** Spray Imidacloprid 0.5ml/litre or neem water\n"
                "- **Whitefly:** Yellow sticky traps + Thiamethoxam 0.3g/litre\n"
                "- **Stem borer (paddy):** Chlorpyriphos granules at tillering stage\n"
                "- **Bollworm (cotton):** Bt spray (Bacillus thuringiensis) or Spinosad\n"
                "- **Fruit fly:** Methyl eugenol bait traps\n\n"
                "**IPM (Integrated Pest Management):**\n"
                "1. Use resistant varieties\n"
                "2. Set pheromone traps for early detection\n"
                "3. Encourage natural enemies (ladybird beetles eat aphids)\n"
                "4. Chemical spray only as last resort — early morning or late evening")

    # --- Government schemes ---
    if any(k in msg for k in ['scheme', 'subsidy', 'loan', 'insurance', 'pm kisan', 'fasal bima', 'yojana', 'sarkar', 'government']):
        return ("**Government Schemes for Farmers** 🏛️\n\n"
                "- **PM-KISAN:** Rs 6,000/year direct benefit — register at pmkisan.gov.in\n"
                "- **PM Fasal Bima Yojana:** Crop insurance at 1.5-2% premium — apply through your bank\n"
                "- **Kisan Credit Card (KCC):** Low-interest loan up to Rs 3 lakh — apply at any bank\n"
                "- **Soil Health Card:** Free soil testing — visit your nearest KVK or block office\n"
                "- **eNAM:** Sell crops online at better prices — enam.gov.in\n"
                "- **PKVY (Paramparagat Krishi):** Rs 50,000/ha grant for organic farming")

    # --- Default helpful response ---
    return ("**FarmAI Offline Assistant** 🌾\n\n"
            "I can help you with:\n"
            "- **Crop diseases** — ask 'my tomato has blight' or use the Plant Disease tab\n"
            "- **Soil & fertilizers** — ask 'what NPK for rice' or 'soil pH'\n"
            "- **Irrigation** — ask 'how much water for wheat'\n"
            "- **Cattle health** — ask 'cow has FMD' or use the Cow Disease tab\n"
            "- **Pest control** — ask 'aphids on my crop'\n"
            "- **Government schemes** — ask 'PM Kisan' or 'crop insurance'\n"
            "- **Crop selection** — ask 'which crop to grow in rabi season'\n\n"
            "_For full AI-powered multilingual chat, a Gemini API key is needed._\n"
            "_Get a free key at: **aistudio.google.com/app/apikey**_")



@app.route('/api/chat-languages')
def chat_languages():
    """Return supported languages list"""
    return jsonify([
        {'code': code, 'name': info['name'], 'greeting': info['greeting']}
        for code, info in SUPPORTED_LANGUAGES.items()
    ])


if __name__ == '__main__':
    app.run(debug=False, port=5000)

