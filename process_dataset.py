import os
import cv2
import numpy as np
import mediapipe as mp
import json
import logging
import sys
import traceback
from datetime import datetime

# Set up logging to file only
log_file = f'dataset_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w'
)

# Also print to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def process_letter(letter_dir, letter_path, hands, arabic_letters):
    letter_data = []
    letter_labels = []
    total = 0
    success = 0
    failed = 0
    
    try:
        images = os.listdir(letter_path)
        for img_file in images:
            total += 1
            img_path = os.path.join(letter_path, img_file)
            
            try:
                # Read and process image
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Could not read image: {img_path}")
                    failed += 1
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
                        
                        letter_data.append(landmarks)
                        letter_labels.append(arabic_letters[letter_dir])
                        success += 1
                else:
                    logging.warning(f"No hand landmarks detected in: {img_path}")
                    failed += 1
                    
            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")
                failed += 1
                continue
    except Exception as e:
        logging.error(f"Error processing letter directory {letter_dir}: {str(e)}")
        traceback.print_exc()
    
    return letter_data, letter_labels, total, success, failed

def process_dataset():
    print("\nStarting dataset processing...")
    logging.info("Starting dataset processing...")
    
    # Initialize MediaPipe Hands
    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Create directories for processed data
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
        print("Created processed_data directory")
    
    # Process each image in the dataset
    dataset_path = 'RGB ArSL dataset'
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found: {dataset_path}")
        return None, None
    
    print(f"Found dataset directory: {dataset_path}")
    
    # Arabic letters mapping
    arabic_letters = {
        'Alef': 0, 'Beh': 1, 'Teh': 2, 'Theh': 3, 'Jeem': 4,
        'Hah': 5, 'Khah': 6, 'Dal': 7, 'Thal': 8, 'Reh': 9,
        'Zain': 10, 'Seen': 11, 'Sheen': 12, 'Sad': 13, 'Dad': 14,
        'Tah': 15, 'Zah': 16, 'Ain': 17, 'Ghain': 18, 'Feh': 19,
        'Qaf': 20, 'Kaf': 21, 'Lam': 22, 'Meem': 23, 'Noon': 24,
        'Heh': 25, 'Waw': 26, 'Yeh': 27
    }
    
    # Get list of letter directories
    letter_dirs = [d for d in os.listdir(dataset_path) if d in arabic_letters]
    print(f"\nFound {len(letter_dirs)} letter directories to process")
    
    all_data = []
    all_labels = []
    total_images = 0
    total_success = 0
    total_failed = 0
    
    for letter_dir in letter_dirs:
        letter_path = os.path.join(dataset_path, letter_dir)
        print(f"\nProcessing letter: {letter_dir}")
        
        try:
            images = os.listdir(letter_path)
            letter_total = len(images)
            print(f"Found {letter_total} images")
            
            letter_success = 0
            letter_failed = 0
            
            for i, img_file in enumerate(images, 1):
                if i % 10 == 0:
                    print(f"Progress: {i}/{letter_total} images", end='\r')
                
                img_path = os.path.join(letter_path, img_file)
                
                try:
                    # Read and process image
                    img = cv2.imread(img_path)
                    if img is None:
                        letter_failed += 1
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
                            
                            all_data.append(landmarks)
                            all_labels.append(arabic_letters[letter_dir])
                            letter_success += 1
                    else:
                        letter_failed += 1
                        
                except Exception as e:
                    letter_failed += 1
                    logging.error(f"Error processing {img_path}: {str(e)}")
                    continue
            
            print(f"\nLetter {letter_dir} complete - Success: {letter_success}, Failed: {letter_failed}")
            total_images += letter_total
            total_success += letter_success
            total_failed += letter_failed
            
        except Exception as e:
            print(f"Error processing letter directory {letter_dir}: {str(e)}")
            logging.error(f"Error processing letter directory {letter_dir}: {str(e)}")
            continue
    
    print(f"""
Processing complete. Statistics:
Total images processed: {total_images}
Successfully processed images: {total_success}
Failed images: {total_failed}
""")
    
    if not all_data:
        print("No data was processed successfully!")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Save processed data
    np.save('processed_data/X.npy', X)
    np.save('processed_data/y.npy', y)
    
    # Save the mapping
    with open('processed_data/letter_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(arabic_letters, f, ensure_ascii=False)
    
    print(f"\nSaved processed data with {len(X)} samples")
    return X, y

if __name__ == "__main__":
    try:
        print("Starting script...")
        X, y = process_dataset()
        if X is not None and y is not None:
            print("Dataset processing completed successfully!")
        else:
            print("Dataset processing failed!")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc() 