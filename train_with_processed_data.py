import numpy as np
from sklearn.model_selection import train_test_split
from model import GestureRecognitionModel
import matplotlib.pyplot as plt
import os

def load_processed_data():
    # Load processed data
    X = np.load('processed_data/X.npy')
    y = np.load('processed_data/y.npy')
    
    print(f"Loaded {len(X)} samples")
    print(f"Number of features per sample: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y

def train_model():
    # Load processed data
    X, y = load_processed_data()
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Initialize and train the model
    model = GestureRecognitionModel()
    
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Save the trained model
    model.save_model()
    print("\nModel saved as 'gesture_model.h5'")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Could not create training history plot: {str(e)}")

if __name__ == "__main__":
    train_model() 