import numpy as np
from model import GestureRecognitionModel
from download_dataset import download_kaggle_dataset, process_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def train_model_with_dataset():
    # Download and process dataset
    if not os.path.exists('processed_data/X.npy'):
        if download_kaggle_dataset():
            X, y = process_dataset()
        else:
            print("Failed to download dataset. Please check your Kaggle API credentials.")
            return
    else:
        # Load processed data
        X = np.load('processed_data/X.npy')
        y = np.load('processed_data/y.npy')
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = GestureRecognitionModel()
    
    print("\nTraining the model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Save the trained model
    model.save_model()
    print("\nModel trained and saved successfully!")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
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

if __name__ == "__main__":
    train_model_with_dataset() 