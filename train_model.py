"""
Plant Disease Detection CNN Training Script
Uses PlantVillage dataset with Transfer Learning (MobileNetV2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json

print("="*70)
print("PLANT DISEASE DETECTION CNN - TRAINING")
print("="*70)

# Configuration
DATASET_PATH = 'Plant Village Dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 20

# Verify dataset
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
    print(f"Please ensure the PlantVillage dataset is extracted to this path")
    exit(1)

# Load disease classes
train_path = os.path.join(DATASET_PATH, 'Train')
disease_classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
NUM_CLASSES = len(disease_classes)

print(f"\n✓ Dataset found!")
print(f"  Location: {DATASET_PATH}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"\nDisease Classes:")
for i, disease in enumerate(disease_classes, 1):
    train_count = len(os.listdir(os.path.join(train_path, disease)))
    print(f"  {i:2d}. {disease:45s} - {train_count} images")

# Setup data generators
print(f"\n{'='*70}")
print("LOADING DATA")
print(f"{'='*70}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Test'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n✓ Data loaded!")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# Build model with Transfer Learning
print(f"\n{'='*70}")
print("BUILDING MODEL")
print(f"{'='*70}")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

print("\n✓ Model built successfully!")
print(f"  Total parameters: {model.count_params():,}")

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create models directory
os.makedirs('models', exist_ok=True)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('models/best_disease_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train Phase 1
print(f"\n{'='*70}")
print("TRAINING PHASE 1: Base Model Frozen")
print(f"{'='*70}")

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Fine-tune
print(f"\n{'='*70}")
print("TRAINING PHASE 2: Fine-tuning Last Layers")
print(f"{'='*70}")

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print(f"\n{'='*70}")
print("EVALUATING MODEL")
print(f"{'='*70}")

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Get predictions
Y_pred = model.predict(test_generator, verbose=0)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

print(f"\n{'-'*70}")
print("CLASSIFICATION REPORT")
print(f"{'-'*70}")
print(classification_report(y_true, y_pred, target_names=disease_classes))

# Save model and metadata
print(f"\n{'='*70}")
print("SAVING MODEL")
print(f"{'='*70}")

model.save('models/plant_disease_model.h5')
model.save('models/plant_disease_model_full/')

with open('models/disease_classes.pkl', 'wb') as f:
    pickle.dump(disease_classes, f)

with open('models/class_indices.pkl', 'wb') as f:
    pickle.dump(train_generator.class_indices, f)

model_info = {
    'model_name': 'PlantVillage Disease Detection',
    'num_classes': NUM_CLASSES,
    'classes': disease_classes,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'image_size': IMAGE_SIZE,
    'architecture': 'MobileNetV2 with Transfer Learning'
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"\n✓ Model saved successfully!")
print(f"  - models/plant_disease_model.h5")
print(f"  - models/plant_disease_model_full/")
print(f"  - models/disease_classes.pkl")
print(f"  - models/class_indices.pkl")
print(f"  - models/model_info.json")
print(f"  - models/best_disease_model.h5")

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Classes: {NUM_CLASSES}")
print(f"✓ Model Parameters: {model.count_params():,}")
print(f"✓ Ready for deployment!")
print(f"{'='*70}\n")
