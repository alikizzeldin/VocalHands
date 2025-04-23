import os
import kaggle
import zipfile
import shutil
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

def download_kaggle_dataset():
    # Set up Kaggle API credentials
    # You need to have kaggle.json in ~/.kaggle/ directory
    try:
        # Download the Arabic Sign Language dataset
        kaggle.api.dataset_download_files(
            'ahmedkhanak1995/arabic-sign-language-dataset',
            path='dataset',
            unzip=True
        )
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def process_dataset():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Create directories for processed data
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    # Process each image in the dataset
    dataset_path = 'dataset/Arabic Sign Language Dataset'
    processed_data = []
    labels = []
    
    # Arabic letters mapping
    arabic_letters = {
        'أ': 0, 'ب': 1, 'ت': 2, 'ث': 3, 'ج': 4,
        'ح': 5, 'خ': 6, 'د': 7, 'ذ': 8, 'ر': 9,
        'ز': 10, 'س': 11, 'ش': 12, 'ص': 13, 'ض': 14,
        'ط': 15, 'ظ': 16, 'ع': 17, 'غ': 18, 'ف': 19,
        'ق': 20, 'ك': 21, 'ل': 22, 'م': 23, 'ن': 24,
        'ه': 25, 'و': 26, 'ي': 27
    }
    
    print("Processing dataset...")
    for letter_dir in tqdm(os.listdir(dataset_path)):
        if letter_dir in arabic_letters:
            letter_path = os.path.join(dataset_path, letter_dir)
            for img_file in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_file)
                
                # Read and process image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract landmarks
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        
                        processed_data.append(landmarks)
                        labels.append(arabic_letters[letter_dir])
    
    # Convert to numpy arrays
    X = np.array(processed_data)
    y = np.array(labels)
    
    # Save processed data
    np.save('processed_data/X.npy', X)
    np.save('processed_data/y.npy', y)
    
    print(f"Processed {len(X)} samples")
    return X, y

if __name__ == "__main__":
    if download_kaggle_dataset():
        X, y = process_dataset()
        print("Dataset processing completed successfully!") 