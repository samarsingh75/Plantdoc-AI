import os
import tensorflow as tf
from model import build_model, compile_model
from preprocess import load_training_data, load_validation_data

TRAIN_DIR = os.environ.get('TRAIN_DIR', 'data/train')
VALID_DIR = os.environ.get('VALID_DIR', 'data/valid')
EPOCHS = int(os.environ.get('EPOCHS', 50))
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', 'saved_model/plant_disease_model.h5')

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


def train():
    training_set = load_training_data(TRAIN_DIR)
    validation_set = load_validation_data(VALID_DIR)

    model = build_model(num_classes=38)
    model = compile_model(model)
    model.summary()

    training_history = model.fit(
        x=training_set,
        validation_data=validation_set,
        epochs=EPOCHS,
        validation_steps=len(validation_set),
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    train_loss, train_acc = model.evaluate(training_set, verbose=1)
    val_loss, val_acc = model.evaluate(validation_set, verbose=1)
    print(f'Training accuracy: {train_acc:.4f}')
    print(f'Validation accuracy: {val_acc:.4f}')

    return training_history


if __name__ == '__main__':
    train()
