import cv2
import numpy as np
import mediapipe as mp
from model import GestureRecognitionModel
import json
from gtts import gTTS
from playsound import playsound
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import logging
import sys
import traceback
import random
from tkinter.font import Font
import webbrowser
from pathlib import Path
import math

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

class ThemeColors:
    def __init__(self):
        # Modern color palette
        self.primary = "#4361ee"        # Primary blue
        self.primary_dark = "#3a56d4"   # Darker blue for hover states
        self.secondary = "#7209b7"      # Purple accent
        self.success = "#06d6a0"        # Green for success indicators
        self.warning = "#ffd166"        # Yellow for warnings
        self.danger = "#ef476f"         # Red for errors/alerts
        self.light = "#f8f9fa"          # Light background
        self.dark = "#212529"           # Dark text
        self.gray = "#6c757d"           # Gray for subtle elements
        self.white = "#ffffff"          # White
        
        # Gradient backgrounds (can be used with Canvas)
        self.gradient_primary = [self.primary, "#4895ef"]  # Blue gradient
        self.gradient_secondary = [self.secondary, "#560bad"]  # Purple gradient
        
    def get_style(self):
        """Configure ttk styles with our theme colors"""
        style = ttk.Style()
        
        # Configure TButton style
        style.configure("TButton",
                      padding=10,
                      font=("Segoe UI", 10))
        
        # Create Accent.TButton for secondary buttons
        style.configure(
            "Accent.TButton",
            background="black",
            foreground="white",
            font=("Segoe UI", 11, "bold")
        )
        
        # Map the button colors for Accent.TButton
        style.map(
            "Accent.TButton",
            background=[("active", "black")],
            foreground=[("active", "white")]
        )
        
        # Create Success.TButton for positive action buttons
        style.configure(
            "Success.TButton",
            background=self.success,
            foreground=self.white,
            font=("Segoe UI", 11, "bold")
        )
        
        # Configure TLabel style for titles
        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 18, "bold"),
            foreground=self.primary
        )
        
        # Configure TLabel style for subtitles
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 14, "bold"),
            foreground=self.secondary
        )
        
        # Configure TFrame style
        style.configure(
            "Card.TFrame",
            background=self.white
        )
        
        # Configure Black.TButton style
        style.configure("Black.TButton",
                      background="black",
                      foreground="white",
                      font=("Segoe UI", 11, "bold"))
        
        style.map("Black.TButton",
                 background=[("active", "black")],
                 foreground=[("active", "white")])
        
        return style

class AnimationUtils:
    @staticmethod
    def fade_in(widget, duration=500, steps=20):
        """Create fade-in animation for widgets"""
        step_time = int(duration / steps)
        for step in range(steps+1):
            alpha = step / steps
            widget.attributes("-alpha", alpha)
            widget.update()
            time.sleep(step_time/1000)
    
    @staticmethod
    def slide_in(widget, start_x, end_x, duration=500, steps=20):
        """Slide a widget from start_x to end_x"""
        step_time = int(duration / steps)
        delta_x = (end_x - start_x) / steps
        current_x = start_x
        for _ in range(steps+1):
            widget.place(x=int(current_x))
            current_x += delta_x
            widget.update()
            time.sleep(step_time/1000)

class IconProvider:
    def __init__(self):
        # Create directory for icons if it doesn't exist
        self.icon_dir = Path("icons")
        if not self.icon_dir.exists():
            self.icon_dir.mkdir(parents=True)
        
        # Define icons using unicode characters with a fallback
        self.icons = {
            'settings': '⚙️',
            'info': 'ℹ️',
            'warning': '⚠️',
            'home': '🏠',
            'hand': '✋',
            'mic': '🎤',
            'translate': '🔄',
            'exit': '🚪',
            'camera': '📷',
            'volume': '🔊',
            'mute': '🔇',
            'loading': '⌛'
        }
    
    def get_icon(self, name):
        """Return icon character for the given name"""
        return self.icons.get(name, '●')  # Default icon if not found

class ArabicLettersTranslator:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title("VocalHands - Arabic Letters Translator")
            self.root.geometry("1024x768")
            self.root.configure(bg="#f8f9fa")
            
            # Initialize theme and icons
            self.theme = ThemeColors()
            self.theme.get_style()
            self.icons = IconProvider()
            
            # Set app icon if available
            try:
                app_icon = tk.PhotoImage(file="VocalHands Logo.png")
                self.root.iconphoto(True, app_icon)
            except:
                logging.warning("App icon not found")
            
            logging.info("Initializing ArabicLettersTranslator...")
            
            # Arabic letters with their names
            self.arabic_letters = {
                "Alif": "أ",
                "Ba": "ب",
                "Ta": "ت",
                "Tha": "ث",
                "Jeem": "ج",
                "Ha": "ح",
                "Kha": "خ",
                "Dal": "د",
                "Dhal": "ذ",
                "Ra": "ر",
                "Zay": "ز",
                "Seen": "س",
                "Sheen": "ش",
                "Sad": "ص",
                "Dad": "ض",
                "Ta": "ط",
                "Dha": "ظ",
                "Ayn": "ع",
                "Ghayn": "غ",
                "Fa": "ف",
                "Qaf": "ق",
                "Kaf": "ك",
                "Lam": "ل",
                "Meem": "م",
                "Noon": "ن",
                "Ha": "ه",
                "Waw": "و",
                "Ya": "ي"
            }
            
            # Create temp directory if it doesn't exist
            self.temp_dir = "temp_audio"
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            
            # Initialize components
            self.setup_components()
            self.setup_webcam()
            self.setup_model()
            
            # Initialize prediction variables
            self.last_prediction = None
            self.confidence_threshold = 0.70
            
            # Start the video capture thread
            self.running = True
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.daemon = True
            self.thread.start()
            
            logging.info("Initialization complete.")
            
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
            sys.exit(1)
    
    def setup_components(self):
        try:
            # Create header frame
            header_frame = ttk.Frame(self.root, style="Card.TFrame")
            header_frame.pack(fill=tk.X, pady=(0, 10))
            
            # App logo and title
            logo_label = ttk.Label(header_frame, text=f"{self.icons.get_icon('translate')} VocalHands - Arabic Letters", 
                                  font=("Segoe UI", 22, "bold"), foreground=self.theme.secondary)
            logo_label.pack(side=tk.LEFT, padx=20, pady=10)
            
            # Create main content frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Create left panel for video
            video_panel = ttk.Frame(main_frame, style="Card.TFrame")
            video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            # Video title
            video_title = ttk.Label(video_panel, text="Camera Feed", style="Subtitle.TLabel")
            video_title.pack(pady=10)
            
            # Create a canvas for video with border
            video_container = ttk.Frame(video_panel, borderwidth=2, relief="ridge")
            video_container.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
            
            self.video_frame = ttk.Label(video_container)
            self.video_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Create right panel for controls and letter display
            control_panel = ttk.Frame(main_frame, width=350)
            control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            control_panel.pack_propagate(False)
            
            # Status frame
            status_frame = ttk.Frame(control_panel, style="Card.TFrame")
            status_frame.pack(fill=tk.X, pady=10)
            
            self.status_indicator = tk.Canvas(status_frame, width=20, height=20, 
                                            bg=self.theme.light, highlightthickness=0)
            self.status_indicator.pack(side=tk.LEFT, padx=10, pady=10)
            self.draw_status_indicator("ready")
            
            self.status_label = ttk.Label(status_frame, text="Status: Ready", font=('Segoe UI', 11))
            self.status_label.pack(side=tk.LEFT, padx=5, pady=10)
            
            # Letter display
            letter_frame = ttk.Frame(control_panel, style="Card.TFrame")
            letter_frame.pack(fill=tk.X, pady=10, padx=5)
            
            letter_title = ttk.Label(letter_frame, text="Detected Letter", style="Subtitle.TLabel")
            letter_title.pack(pady=(10, 5))
            
            letter_card = ttk.Frame(letter_frame, relief="ridge", borderwidth=2)
            letter_card.pack(fill=tk.X, padx=15, pady=10)
            
            self.letter_label = ttk.Label(letter_card, text="Waiting for gesture...", 
                                        font=('Segoe UI', 14, 'bold'), 
                                        foreground=self.theme.dark)
            self.letter_label.pack(pady=(15, 8), padx=10)
            
            self.arabic_label = ttk.Label(letter_card, text="", 
                                        font=('Segoe UI', 22, 'bold'), 
                                        foreground=self.theme.secondary)
            self.arabic_label.pack(pady=(0, 15), padx=10)
            
            # Buttons frame
            buttons_frame = ttk.Frame(control_panel)
            buttons_frame.pack(fill=tk.X, pady=20)
            
            # Basic words button
            basic_words_btn = ttk.Button(buttons_frame, text=f"{self.icons.get_icon('hand')} Basic Words", 
                                        style="Black.TButton", command=self.go_to_basic_words)
            basic_words_btn.pack(fill=tk.X, pady=5)
            
            # Exit button
            exit_btn = ttk.Button(buttons_frame, text=f"{self.icons.get_icon('exit')} Exit Application", 
                                 command=self.exit_program)
            exit_btn.pack(fill=tk.X, pady=5)
            
            logging.info("GUI components set up successfully.")
            
        except Exception as e:
            logging.error(f"Error setting up components: {str(e)}")
            raise
    
    def setup_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            # Restore original resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            logging.info("Webcam initialized successfully.")
            self.status_label.config(text="Status: Camera Ready")
            
        except Exception as e:
            logging.error(f"Error setting up webcam: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize webcam: {str(e)}")
            raise
    
    def setup_model(self):
        try:
            from model import GestureRecognitionModel
            self.model = GestureRecognitionModel()
            model_path = 'gesture_model.h5'
            
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
                
            self.model.load_model(model_path)
            
            # Initialize prediction variables
            self.last_prediction_time = 0
            self.prediction_cooldown = 0.1
            self.current_confidence = 0
            
            # Initialize gesture buffer for stable predictions
            self.gesture_buffer = []
            self.buffer_size = 5
            
            logging.info("Model loaded successfully.")
            self.status_label.config(text="Status: Ready to Translate")
            
        except Exception as e:
            logging.error(f"Error setting up model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            raise
    
    def update_frame(self):
        last_prediction_time = 0
        prediction_cooldown = 0.1
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    continue
                    
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                current_time = time.time()
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                # Add stylish border and detection status
                border_color = (114, 9, 183)  # BGR format for secondary purple
                border_thickness = 5
                
                # Top border with text
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), border_color, -1)
                cv2.putText(display_frame, "Arabic Letters Translator", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Detection status text
                detection_text = "Hand Detected" if results.multi_hand_landmarks else "No Hand Detected"
                detection_color = (6, 214, 160) if results.multi_hand_landmarks else (239, 71, 111)
                cv2.putText(display_frame, detection_text, 
                           (display_frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw border
                cv2.rectangle(display_frame, (0, 0), 
                             (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                             border_color, border_thickness)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.draw_stylized_hand_landmarks(display_frame, hand_landmarks)
                        
                        # Only predict if enough time has passed since last prediction
                        if current_time - last_prediction_time >= prediction_cooldown:
                            # Predict gesture
                            gesture_id, confidence, predicted_letter = self.model.predict_gesture(hand_landmarks)
                            
                            # Update current confidence
                            self.current_confidence = confidence
                            
                            # Add to buffer
                            self.gesture_buffer.append(gesture_id)
                            if len(self.gesture_buffer) > self.buffer_size:
                                self.gesture_buffer.pop(0)
                            
                            # Get most common gesture in buffer
                            if len(self.gesture_buffer) == self.buffer_size:
                                counts = np.bincount(self.gesture_buffer)
                                most_common = np.argmax(counts)
                                if counts[most_common] >= self.buffer_size * 0.6 and confidence >= self.confidence_threshold:
                                    # Only update if prediction has changed
                                    if predicted_letter != self.last_prediction:
                                        # Get Arabic letter
                                        arabic_letter = self.arabic_letters.get(predicted_letter, predicted_letter)
                                        
                                        # Update labels
                                        self.letter_label.config(text=f"English: {predicted_letter}")
                                        self.arabic_label.config(text=f"{arabic_letter}")
                                        
                                        # Speak the letter
                                        self.speak_arabic(arabic_letter)
                                        
                                        self.last_prediction = predicted_letter
                            
                            last_prediction_time = current_time
                else:
                    # Reset buffer when no hand is detected
                    self.gesture_buffer = []
                    self.current_confidence = 0
                
                # Resize for display
                display_h, display_w = 480, 640
                h, w = display_frame.shape[:2]
                
                # Calculate scaling factor
                scale = min(display_w / w, display_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize image
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                
                # Create black padding
                padded_frame = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                
                # Calculate offsets for centering
                x_offset = (display_w - new_w) // 2
                y_offset = (display_h - new_h) // 2
                
                # Place the resized image on the padded frame
                padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = display_frame
                
                # Convert to PhotoImage
                image = Image.fromarray(cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(image=image)
                
                # Update video frame
                self.video_frame.config(image=photo)
                self.video_frame.image = photo
                
            except Exception as e:
                logging.error(f"Error in update_frame: {str(e)}")
                self.status_label.config(text=f"Status: Error - {str(e)}")
                time.sleep(1)
                continue
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    def draw_stylized_hand_landmarks(self, image, hand_landmarks):
        """Draw stylized hand landmarks with better visibility"""
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = self.mp_hands
        
        # Custom drawing specs
        landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color for landmarks
            thickness=5,
            circle_radius=5
        )
        connection_drawing_spec = mp_drawing.DrawingSpec(
            color=(255, 255, 255),  # White color for connections
            thickness=3
        )
        
        # Draw the landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec
        )
    
    def speak_arabic(self, text):
        try:
            # Create a unique temporary file name
            temp_file = os.path.join(self.temp_dir, f"speech_{time.time()}.mp3")
            
            # Generate speech using gTTS with faster speed
            tts = gTTS(text=text, lang='ar', slow=False)
            tts.save(temp_file)
            
            # Play the audio in a separate thread
            def play_and_cleanup():
                try:
                    playsound(temp_file)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            
            # Start playback in a separate thread
            threading.Thread(target=play_and_cleanup, daemon=True).start()
            
        except Exception as e:
            logging.error(f"Error in text-to-speech: {str(e)}")
    
    def go_to_basic_words(self):
        try:
            self.running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
            
            # Destroy current window
            self.root.destroy()
            
            # Create new window for basic words translator
            new_root = tk.Tk()
            app = BasicWordsTranslator(new_root)
            new_root.mainloop()
            
        except Exception as e:
            logging.error(f"Error switching to basic words mode: {str(e)}")
            messagebox.showerror("Error", f"Failed to switch to basic words mode: {str(e)}")
    
    def exit_program(self):
        try:
            self.running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(self.temp_dir)
                except:
                    pass
            
            self.root.quit()
            self.root.destroy()
            logging.info("Program terminated successfully.")
        except Exception as e:
            logging.error(f"Error during exit: {str(e)}")
            sys.exit(1)

    def draw_status_indicator(self, status):
        """Draw a status indicator with animations"""
        self.status_indicator.delete("all")
        
        if status == "ready":
            # Green circle for ready
            self.status_indicator.create_oval(2, 2, 18, 18, fill=self.theme.success, outline="")
        elif status == "processing":
            # Yellow spinning indicator
            angle = (self.animation_frame % 12) * 30
            self.status_indicator.create_arc(2, 2, 18, 18, start=angle, extent=120, 
                                            fill=self.theme.warning, outline="", style=tk.PIESLICE)
        elif status == "error":
            # Red X for error
            self.status_indicator.create_oval(2, 2, 18, 18, fill=self.theme.danger, outline="")
            self.status_indicator.create_line(6, 6, 14, 14, fill=self.theme.white, width=2)
            self.status_indicator.create_line(14, 6, 6, 14, fill=self.theme.white, width=2)

class BasicWordsTranslator:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title("VocalHands - Arabic Basic Words Translator")
            self.root.geometry("1024x768")
            self.root.configure(bg="#f8f9fa")
            
            # Initialize theme and icons
            self.theme = ThemeColors()
            self.theme.get_style()
            self.icons = IconProvider()
            
            # Set app icon if available
            try:
                app_icon = tk.PhotoImage(file="VocalHands Logo.png")
                self.root.iconphoto(True, app_icon)
            except:
                logging.warning("App icon not found")
            
            logging.info("Initializing BasicWordsTranslator...")
            
            # Arabic translations with phonetic text for better pronunciation
            self.arabic_translations = {
                "Thanks": "شُكراً",
                "Sorry": "عُذراً",
                "Salam aleikum": "السَّلامُ عَلَيكُم",
                "Not bad": "لَيسَ سَيِّئاً",
                "I am sorry": "أَنا آسِف",
                "I am pleased to meet you": "سَعيدٌ بِلِقائِك",
                "I am fine": "أَنا بِخَير",
                "How are you": "كَيفَ حالُك",
                "Good morning": "صَباحُ الخَير",
                "Good evening": "مَساءُ الخَير",
                "Good bye": "مَعَ السَّلامَة",
                "Alhamdulillah": "الحَمدُ لِله"
            }
            
            # Create visual representations of phrases for display
            self.phrase_colors = {}
            self.phrase_animations = {}
            
            # Assign unique colors to each phrase
            colors = [
                "#4361ee",  # Blue
                "#3a0ca3",  # Deep Purple
                "#7209b7",  # Purple
                "#f72585",  # Pink
                "#4cc9f0",  # Light Blue
                "#06d6a0",  # Green
                "#ff5e5b",  # Coral
                "#ffbe0b",  # Yellow
                "#ff7b00",  # Orange
                "#8338ec",  # Violet
                "#fb5607",  # Red-Orange
                "#277da1",  # Teal
            ]
            
            for i, phrase in enumerate(self.arabic_translations.keys()):
                self.phrase_colors[phrase] = colors[i % len(colors)]
                # Initialize animation values (scale, rotation)
                self.phrase_animations[phrase] = {"scale": 1.0, "rotation": 0}
            
            # Create temp directory if it doesn't exist
            self.temp_dir = "temp_audio"
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            
            # Initialize components
            self.setup_components()
            self.setup_webcam()
            self.setup_model()
            
            # Initialize prediction variables
            self.last_prediction = None
            self.confidence_threshold = 0.70
            
            # Animation variables
            self.animation_active = False
            self.animation_frame = 0
            self.current_phrase = None
            self.confidence_display = 0
            
            # Start the video capture thread
            self.running = True
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.daemon = True
            self.thread.start()
            
            # Start the animation thread
            self.animation_thread = threading.Thread(target=self.update_animations)
            self.animation_thread.daemon = True
            self.animation_thread.start()
            
            logging.info("Initialization complete.")
            
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
            sys.exit(1)
    
    def setup_components(self):
        try:
            # Create header frame
            header_frame = ttk.Frame(self.root, style="Card.TFrame")
            header_frame.pack(fill=tk.X, pady=(0, 10))
            
            # App logo and title
            logo_label = ttk.Label(header_frame, text=f"{self.icons.get_icon('translate')} VocalHands - Basic Words", 
                                  font=("Segoe UI", 22, "bold"), foreground=self.theme.secondary)
            logo_label.pack(side=tk.LEFT, padx=20, pady=10)
            
            # Create main content frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Create left panel for video
            video_panel = ttk.Frame(main_frame, style="Card.TFrame")
            video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
            # Video title
            video_title = ttk.Label(video_panel, text="Camera Feed", style="Subtitle.TLabel")
            video_title.pack(pady=10)
            
            # Create a canvas for video with border
            video_container = ttk.Frame(video_panel, borderwidth=2, relief="ridge")
            video_container.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
            
            self.video_frame = ttk.Label(video_container)
            self.video_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Confidence meter below video
            confidence_frame = ttk.Frame(video_panel)
            confidence_frame.pack(fill=tk.X, padx=20, pady=10)
            
            ttk.Label(confidence_frame, text="Recognition Confidence:").pack(side=tk.LEFT)
            
            # Canvas for confidence meter
            self.confidence_canvas = tk.Canvas(confidence_frame, height=20, width=300, 
                                              bg=self.theme.light, highlightthickness=0)
            self.confidence_canvas.pack(side=tk.LEFT, padx=10)
            self.update_confidence_meter(0)
            
            # Create right panel for controls and word display
            control_panel = ttk.Frame(main_frame, width=350)
            control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            control_panel.pack_propagate(False)
            
            # Status frame with animated indicator
            status_frame = ttk.Frame(control_panel, style="Card.TFrame")
            status_frame.pack(fill=tk.X, pady=10)
            
            self.status_indicator = tk.Canvas(status_frame, width=20, height=20, 
                                            bg=self.theme.light, highlightthickness=0)
            self.status_indicator.pack(side=tk.LEFT, padx=10, pady=10)
            self.draw_status_indicator("ready")
            
            self.status_label = ttk.Label(status_frame, text="Status: Ready", font=('Segoe UI', 11))
            self.status_label.pack(side=tk.LEFT, padx=5, pady=10)
            
            # Result display with animation
            result_frame = ttk.Frame(control_panel, style="Card.TFrame")
            result_frame.pack(fill=tk.X, pady=10, padx=5)
            
            # Word display card
            result_title = ttk.Label(result_frame, text="Detected Word", style="Subtitle.TLabel")
            result_title.pack(pady=(10, 5))
            
            # Create a nice card for the word display
            word_card = ttk.Frame(result_frame, relief="ridge", borderwidth=2)
            word_card.pack(fill=tk.X, padx=15, pady=10)
            
            # English word label
            self.word_label = ttk.Label(word_card, text="Waiting for gesture...", 
                                      font=('Segoe UI', 14, 'bold'), 
                                      foreground=self.theme.dark)
            self.word_label.pack(pady=(15, 8), padx=10)
            
            # Arabic word label with larger font
            self.arabic_label = ttk.Label(word_card, text="", 
                                        font=('Segoe UI', 22, 'bold'), 
                                        foreground=self.theme.secondary)
            self.arabic_label.pack(pady=(0, 15), padx=10)
            
            # Buttons frame
            buttons_frame = ttk.Frame(control_panel)
            buttons_frame.pack(fill=tk.X, pady=20)
            
            # Arabic letters button
            letters_btn = ttk.Button(buttons_frame, text=f"{self.icons.get_icon('hand')} Arabic Letters", 
                                    style="Black.TButton", command=self.go_to_arabic_letters)
            letters_btn.pack(fill=tk.X, pady=5)
            
            # Exit button
            exit_btn = ttk.Button(buttons_frame, text=f"{self.icons.get_icon('exit')} Exit Application", 
                                 command=self.exit_program)
            exit_btn.pack(fill=tk.X, pady=5)
            
            logging.info("GUI components set up successfully.")
            
        except Exception as e:
            logging.error(f"Error setting up components: {str(e)}")
            raise
    
    def setup_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            # Restore original resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            logging.info("Webcam initialized successfully.")
            self.status_label.config(text="Status: Camera Ready")
            
        except Exception as e:
            logging.error(f"Error setting up webcam: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize webcam: {str(e)}")
            raise
    
    def setup_model(self):
        try:
            from model import BasicWordsModel
            self.model = BasicWordsModel()
            model_path = 'basic_words_model.h5'
            
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
                
            self.model.load_model(model_path)
            
            # Initialize prediction variables
            self.last_prediction_time = 0
            self.prediction_cooldown = 0.1
            self.current_confidence = 0
            
            # Initialize gesture buffer for stable predictions
            self.gesture_buffer = []
            self.buffer_size = 5
            
            logging.info("Model loaded successfully.")
            self.status_label.config(text="Status: Ready to Translate")
            
        except Exception as e:
            logging.error(f"Error setting up model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            raise
    
    def draw_status_indicator(self, status):
        """Draw a status indicator with animations"""
        self.status_indicator.delete("all")
        
        if status == "ready":
            # Green circle for ready
            self.status_indicator.create_oval(2, 2, 18, 18, fill=self.theme.success, outline="")
        elif status == "processing":
            # Yellow spinning indicator
            angle = (self.animation_frame % 12) * 30
            self.status_indicator.create_arc(2, 2, 18, 18, start=angle, extent=120, 
                                            fill=self.theme.warning, outline="", style=tk.PIESLICE)
        elif status == "error":
            # Red X for error
            self.status_indicator.create_oval(2, 2, 18, 18, fill=self.theme.danger, outline="")
            self.status_indicator.create_line(6, 6, 14, 14, fill=self.theme.white, width=2)
            self.status_indicator.create_line(14, 6, 6, 14, fill=self.theme.white, width=2)
    
    def update_confidence_meter(self, confidence):
        """Update the confidence meter with animation"""
        self.confidence_canvas.delete("all")
        
        # Background
        self.confidence_canvas.create_rectangle(0, 0, 300, 20, fill=self.theme.light, outline="")
        
        # Calculate width based on confidence
        width = int(confidence * 300)
        
        # Determine color based on confidence
        if confidence < 0.4:
            color = self.theme.danger
        elif confidence < 0.7:
            color = self.theme.warning
        else:
            color = self.theme.success
            
        # Fill bar
        self.confidence_canvas.create_rectangle(0, 0, width, 20, fill=color, outline="")
    
    def animate_phrase_detection(self, phrase):
        """Animate the phrase detection"""
        # Store the current phrase
        self.current_phrase = phrase
        
        # Reset animation
        self.animation_active = True
        self.animation_frame = 0
        
        # Get Arabic translation
        arabic_word = self.arabic_translations.get(phrase, phrase)
        
        # Update GUI in main thread with animation
        self.word_label.config(text=f"English: {phrase}")
        self.arabic_label.config(text=f"{arabic_word}")
        
        # Assign highlight color based on phrase
        phrase_color = self.phrase_colors.get(phrase, self.theme.primary)
        self.word_label.config(foreground=phrase_color)
    
    def update_animations(self):
        """Update all animations in a separate thread"""
        while self.running:
            try:
                # Update animation frame counter
                self.animation_frame += 1
                
                # Update status indicator (spinning when processing)
                if hasattr(self, 'status_indicator'):
                    current_status = "processing" if self.animation_frame % 30 < 15 else "ready"
                    self.draw_status_indicator(current_status)
                
                # Phrase detection animation
                if self.animation_active and self.current_phrase and hasattr(self, 'word_label'):
                    # Get color for current phrase
                    phrase_color = self.phrase_colors.get(self.current_phrase, self.theme.primary)
                    
                    if self.animation_frame < 10:
                        # Scaling animation for text
                        scale = 1.0 + 0.2 * math.sin(self.animation_frame * 0.314)  # Scaled sine wave
                        self.word_label.config(font=('Segoe UI', int(14 * scale), 'bold'))
                        self.arabic_label.config(font=('Segoe UI', int(22 * scale), 'bold'))
                    else:
                        # Animation complete
                        self.animation_active = False
                        self.word_label.config(font=('Segoe UI', 14, 'bold'))
                        self.arabic_label.config(font=('Segoe UI', 22, 'bold'))
                
                # Gradually update confidence display for smoothness
                if hasattr(self, 'confidence_display') and hasattr(self, 'current_confidence'):
                    # Smooth transition
                    self.confidence_display += (self.current_confidence - self.confidence_display) * 0.2
                    if abs(self.confidence_display - self.current_confidence) < 0.01:
                        self.confidence_display = self.current_confidence
                    
                    self.update_confidence_meter(self.confidence_display)
                
                time.sleep(0.05)  # 50ms update rate for animations
                
            except Exception as e:
                logging.error(f"Animation error: {e}")
                time.sleep(0.5)
    
    def update_frame(self):
        last_prediction_time = 0
        prediction_cooldown = 0.1
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    continue
                    
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                current_time = time.time()
                
                # Create a copy for drawing
                display_frame = frame.copy()
                
                # Add stylish border and detection status
                border_color = (114, 9, 183)  # BGR format for secondary purple
                border_thickness = 5
                
                # Top border with text
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), border_color, -1)
                cv2.putText(display_frame, "Basic Words Translator", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Detection status text
                detection_text = "Hand Detected" if results.multi_hand_landmarks else "No Hand Detected"
                detection_color = (6, 214, 160) if results.multi_hand_landmarks else (239, 71, 111)
                cv2.putText(display_frame, detection_text, 
                           (display_frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw border
                cv2.rectangle(display_frame, (0, 0), 
                             (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                             border_color, border_thickness)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.draw_stylized_hand_landmarks(display_frame, hand_landmarks)
                        
                        # Only predict if enough time has passed since last prediction
                        if current_time - last_prediction_time >= prediction_cooldown:
                            # Predict gesture
                            gesture_id, confidence, predicted_word = self.model.predict_gesture(hand_landmarks)
                            
                            # Update current confidence for animation
                            self.current_confidence = confidence
                            
                            # Add to buffer
                            self.gesture_buffer.append(gesture_id)
                            if len(self.gesture_buffer) > self.buffer_size:
                                self.gesture_buffer.pop(0)
                            
                            # Get most common gesture in buffer
                            if len(self.gesture_buffer) == self.buffer_size:
                                counts = np.bincount(self.gesture_buffer)
                                most_common = np.argmax(counts)
                                if counts[most_common] >= self.buffer_size * 0.6 and confidence >= self.confidence_threshold:
                                    # Only update and speak if prediction has changed
                                    if predicted_word != self.last_prediction:
                                        # Get Arabic translation
                                        arabic_word = self.arabic_translations.get(predicted_word, predicted_word)
                                        
                                        # Animate the phrase detection
                                        self.animate_phrase_detection(predicted_word)
                                        
                                        # Speak the Arabic word
                                        self.speak_arabic(arabic_word)
                                        
                                        self.last_prediction = predicted_word
                            
                            last_prediction_time = current_time
                else:
                    # Reset buffer when no hand is detected
                    self.gesture_buffer = []
                    self.current_confidence = 0
                
                # Resize for display
                display_h, display_w = 480, 640
                h, w = display_frame.shape[:2]
                
                # Calculate scaling factor
                scale = min(display_w / w, display_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize image
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                
                # Create black padding
                padded_frame = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                
                # Calculate offsets for centering
                x_offset = (display_w - new_w) // 2
                y_offset = (display_h - new_h) // 2
                
                # Place the resized image on the padded frame
                padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = display_frame
                
                # Convert to PhotoImage
                image = Image.fromarray(cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(image=image)
                
                # Update video frame
                self.video_frame.config(image=photo)
                self.video_frame.image = photo
                
            except Exception as e:
                logging.error(f"Error in update_frame: {str(e)}")
                self.status_label.config(text=f"Status: Error - {str(e)}")
                time.sleep(1)
                continue
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    def draw_stylized_hand_landmarks(self, image, hand_landmarks):
        """Draw stylized hand landmarks with better visibility"""
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = self.mp_hands
        
        # Custom drawing specs
        landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color for landmarks
            thickness=5,
            circle_radius=5
        )
        connection_drawing_spec = mp_drawing.DrawingSpec(
            color=(255, 255, 255),  # White color for connections
            thickness=3
        )
        
        # Draw the landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec
        )
    
    def speak_arabic(self, text):
        try:
            # Create a unique temporary file name
            temp_file = os.path.join(self.temp_dir, f"speech_{time.time()}.mp3")
            
            # Generate speech using gTTS with faster speed
            tts = gTTS(text=text, lang='ar', slow=False)
            tts.save(temp_file)
            
            # Play the audio in a separate thread
            def play_and_cleanup():
                try:
                    playsound(temp_file)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
            
            # Start playback in a separate thread
            threading.Thread(target=play_and_cleanup, daemon=True).start()
            
        except Exception as e:
            logging.error(f"Error in text-to-speech: {str(e)}")
    
    def go_to_arabic_letters(self):
        try:
            self.running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
            
            # Destroy current window
            self.root.destroy()
            
            # Create new window for Arabic letters translator
            new_root = tk.Tk()
            app = ArabicLettersTranslator(new_root)
            new_root.mainloop()
            
        except Exception as e:
            logging.error(f"Error switching to Arabic letters mode: {str(e)}")
            messagebox.showerror("Error", f"Failed to switch to Arabic letters mode: {str(e)}")
    
    def exit_program(self):
        try:
            self.running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            # Clean up temp directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(self.temp_dir)
                except:
                    pass
            
            self.root.quit()
            self.root.destroy()
            logging.info("Program terminated successfully.")
        except Exception as e:
            logging.error(f"Error during exit: {str(e)}")
            sys.exit(1)

def create_splash_screen():
    """Create an animated splash screen on startup"""
    splash_root = tk.Tk()
    splash_root.overrideredirect(True)  # Remove window decorations
    
    # Get screen width and height
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()
    
    # Set size and position
    splash_width, splash_height = 500, 300
    x = (screen_width - splash_width) // 2
    y = (screen_height - splash_height) // 2
    
    splash_root.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
    
    # Configure splash screen
    splash_root.configure(bg="#4361ee")  # Primary blue background
    
    # Create gradient effect using Canvas
    canvas = tk.Canvas(splash_root, width=splash_width, height=splash_height, 
                      highlightthickness=0, bg="#4361ee")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Create gradient
    for i in range(splash_height):
        # Calculate color gradient from blue to purple
        r = int(67 + (114 - 67) * i / splash_height)
        g = int(97 + (9 - 97) * i / splash_height)
        b = int(238 + (183 - 238) * i / splash_height)
        color = f"#{r:02x}{g:02x}{b:02x}"
        canvas.create_line(0, i, splash_width, i, fill=color, width=1)
    
    # App title
    title_text = canvas.create_text(splash_width//2, 80, text="VocalHands",
                                  font=("Segoe UI", 36, "bold"), fill="#ffffff")
    
    # Subtitle
    subtitle_text = canvas.create_text(splash_width//2, 130, 
                                     text="Arabic Sign Language Translator",
                                     font=("Segoe UI", 14), fill="#ffffff")
    
    # Loading animation
    loading_text = canvas.create_text(splash_width//2, splash_height-50, 
                                    text="Loading...", font=("Segoe UI", 10),
                                    fill="#ffffff")
    
    # Progress bar background
    progress_bg = canvas.create_rectangle(50, splash_height-80, splash_width-50, 
                                       splash_height-70, fill="#ffffff", outline="")
    
    # Loading icon
    hand_icon = "✋"  # Hand emoji
    icon_text = canvas.create_text(splash_width//2, splash_height//2, 
                                 text=hand_icon, font=("Segoe UI", 72),
                                 fill="#ffffff")
    
    splash_root.update()
    
    # Animation function
    def animate_splash(step=0, progress=0):
        if step < 20:  # Fade in
            alpha = step / 20
            splash_root.attributes("-alpha", alpha)
        elif step < 40:  # Pulsate icon
            scale = 1.0 + 0.1 * math.sin((step - 20) * 0.314)
            canvas.itemconfig(icon_text, font=("Segoe UI", int(72 * scale)))
        
        # Update progress bar
        progress += random.uniform(1, 5)
        progress = min(100, progress)
        progress_width = (splash_width - 100) * (progress / 100)
        
        # Progress indicator
        canvas.delete("progress_indicator")
        canvas.create_rectangle(50, splash_height-80, 50 + progress_width, 
                              splash_height-70, fill="#06d6a0", outline="",
                              tags="progress_indicator")
        
        # Continue animation or close
        if progress < 100 and step < 100:
            splash_root.after(50, lambda: animate_splash(step + 1, progress))
        else:
            # Final animation - fade out
            for i in range(10, -1, -1):
                splash_root.attributes("-alpha", i/10)
                splash_root.update()
                time.sleep(0.05)
            splash_root.destroy()
    
    # Start animation
    animate_splash()
    
    # Keep splash screen open until animation completes
    splash_root.mainloop()

def main():
    try:
        # Show splash screen
        create_splash_screen()
        
        # Set up main application
        root = tk.Tk()
        
        # Set app theme - make it look modern
        if "win" in sys.platform:
            # Use Windows theme
            root.tk.call("source", "azure.tcl")
            root.tk.call("set_theme", "light")
        
        # Start with Arabic Letters Translator
        app = ArabicLettersTranslator(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 