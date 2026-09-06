from tensorflow.keras.models import load_model
import os

try:
    print("Loading best_disease_model.h5...")
    model = load_model('models/best_disease_model.h5', safe_mode=False)
    print("✓ Model loaded successfully!")
    
    print("Resaving model in TensorFlow format...")
    model.save('models/best_disease_model.h5', save_format='tf')
    print("✓ Model resaved successfully!")
    print("\nYour model is now compatible. Try running 'python app.py' again!")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
