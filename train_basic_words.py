import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from model import BasicWordsModel
import json
from sklearn.model_selection import train_test_split
import sys

def extract_frames_from_video(video_path, max_frames=30):
    """Extract frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval to get evenly distributed frames
    interval = max(1, total_frames // max_frames)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only take frames at the calculated interval
        if frame_count % interval == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
                
        frame_count += 1
    
    cap.release()
    return frames

def process_dataset(dataset_path):
    print(f"\nProcessing dataset from: {dataset_path}")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Initialize lists to store data
    X = []
    y = []
    class_names = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Found {len(class_dirs)} classes to process")
    
    # Process each class
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Processing classes", file=sys.stdout)):
        class_path = os.path.join(dataset_path, class_dir)
        class_name = class_dir
        class_names.append(class_name)
        
        # Process each video in the class directory
        video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
        print(f"\nProcessing {len(video_files)} videos for class: {class_name}")
        
        # Create progress bar for videos in this class
        for video_file in tqdm(video_files, desc=f"Processing {class_name}", leave=True, file=sys.stdout):
            video_path = os.path.join(class_path, video_file)
            
            # Extract frames from video
            frames = extract_frames_from_video(video_path)
            
            if not frames:
                print(f"\nWarning: Could not extract frames from video: {video_path}")
                continue
            
            # Process each frame
            for frame in frames:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract landmarks
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        
                        X.append(landmarks)
                        y.append(class_idx)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nProcessed {len(X)} samples from {len(class_names)} classes")
    print("Class names:", class_names)
    
    # Save class names
    with open('basic_words_classes.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)
    
    return X, y

def main():
    print("Starting Basic Words Model Training")
    print("==================================")
    
    # Paths
    train_path = 'ArSL Basic Words/train'
    val_path = 'ArSL Basic Words/val'
    
    print("\nProcessing training data...")
    X_train, y_train = process_dataset(train_path)
    
    print("\nProcessing validation data...")
    X_val, y_val = process_dataset(val_path)
    
    print("\nCreating and training model...")
    model = BasicWordsModel()
    
    print("\nTraining model...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Save model
    model.save_model()
    
    print("\nTraining complete!")
    print("Model saved as 'basic_words_model.h5'")
    print("Class dictionary saved as 'basic_words_dict.json'")

if __name__ == "__main__":
    main() 