import numpy as np
from model import GestureRecognitionModel, collect_gesture_data
import os

def train_gesture_model():
    # Initialize the model
    model = GestureRecognitionModel()
    
    # Create directories for training data if they don't exist
    if not os.path.exists('training_data'):
        os.makedirs('training_data')
    
    # Collect training data for each gesture
    X_train = []
    y_train = []
    
    print("Starting data collection for training...")
    print("For each gesture, you will need to perform the sign 100 times.")
    print("Press 'q' to stop collecting data for the current gesture.")
    
    for gesture_id, gesture_text in model.gesture_dict.items():
        print(f"\nCollecting data for gesture: {gesture_text}")
        gesture_data = collect_gesture_data(gesture_id)
        
        # Save the collected data
        np.save(f'training_data/gesture_{gesture_id}.npy', gesture_data)
        
        X_train.append(gesture_data)
        y_train.extend([gesture_id] * len(gesture_data))
    
    # Combine all training data
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("\nTraining the model...")
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Save the trained model
    model.save_model()
    print("\nModel trained and saved successfully!")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
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
    except ImportError:
        print("Matplotlib not installed. Skipping training history plot.")

if __name__ == "__main__":
    train_gesture_model() 