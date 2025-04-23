# Arabic Sign Language Translator

This project implements a real-time Arabic sign language translator that can convert hand gestures captured from a webcam or mobile camera into Arabic text and voice output.

## Features

- Real-time hand gesture detection using MediaPipe
- Arabic sign language recognition
- Text display of recognized gestures
- Voice output in Arabic
- Support for multiple gestures

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python sign_language_translator.py
```

2. Make sure your webcam is connected and working
3. Perform sign language gestures in front of the camera
4. The system will display the recognized gesture in Arabic text and speak it out loud
5. Press 'q' to quit the application

## Gestures Currently Supported

- مرحباً (Hello)
- شكراً (Thank you)
- نعم (Yes)
- لا (No)
- أحبك (I love you)

## Requirements

- Python 3.7 or higher
- Webcam or mobile camera
- Internet connection (for text-to-speech functionality)

## Notes

- Make sure you have proper lighting for better gesture recognition
- Keep your hands within the camera frame
- The system works best with clear, distinct gestures 