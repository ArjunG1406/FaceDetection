import sys
import cv2
import threading
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque, Counter
from deepface import DeepFace

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap

# 1. Custom Emotion Model Architecture
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------- CONFIG ----------
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_HEX = {
    "Happy": "#00e5ff", "Sad": "#788cff", "Angry": "#ff3cac",
    "Fear": "#ff9900", "Surprise": "#ffe44d", "Disgust": "#00ff9d", "Neutral": "#8899aa",
}

class Signals(QObject):
    frame_ready = pyqtSignal(object)

# 2. Logic Thread (Asynchronous Dual-Model Engine)
class DetectionApp:
    def __init__(self, signals):
        self.signals = signals
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Emotion Model
        self.model = EmotionCNN(num_classes=7).to(self.device)
        try:
            self.model.load_state_dict(torch.load("models/emotion_model.pth", map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            print("Error: models/emotion_model.pth not found!")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Async Gender Tracking
        self.last_gender = "Analyzing..."
        self.is_analyzing_gender = False 

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_gender_inference(self, face_img):
        """Heavy lifting for gender is done here in a background worker thread."""
        try:
            analysis = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False, silent=True)
            self.last_gender = analysis[0]['dominant_gender']
        except:
            pass
        self.is_analyzing_gender = False

    def _loop(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            display = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Higher sensitivity for better detection
            faces_raw = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            h_disp, w_disp = display.shape[:2]
            current_faces = []

            for i, (x, y, w, h) in enumerate(faces_raw):
                # A. Fast Emotion Detection (Main Loop)
                face_roi_gray = gray[y:y+h, x:x+w]
                input_tensor = self.transform(face_roi_gray).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    pred = torch.max(outputs, 1)[1].item()
                
                emotion = EMOTION_LABELS[pred]

                # B. Async Gender Detection (Worker Thread)
                # If the worker is free, give it a new face to analyze
                if not self.is_analyzing_gender:
                    self.is_analyzing_gender = True
                    face_roi_color = frame[y:y+h, x:x+w]
                    threading.Thread(target=self._run_gender_inference, args=(face_roi_color,), daemon=True).start()

                # C. Visualization using current emotion and last known gender
                fx = w_disp - x - w
                h_color = EMOTION_HEX.get(emotion, "#00e5ff").lstrip('#')
                color_bgr = tuple(int(h_color[i:i+2], 16) for i in (4, 2, 0))

                label_text = f"{self.last_gender} | {emotion}"
                cv2.rectangle(display, (fx, y), (fx+w, y+h), color_bgr, 2)
                cv2.putText(display, label_text, (fx, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

                current_faces.append({
                    "emotion": emotion,
                    "gender": self.last_gender,
                    "confidence": 100 
                })

            self.signals.frame_ready.emit((display, current_faces))
        cap.release()

# 3. Main GUI Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time High-Speed Analytics (30 FPS)")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #050810; color: #e8eaf0; }
            QLabel { color: #e8eaf0; }
            QPushButton {
                font-size: 12px; padding: 12px 28px;
                border: 1px solid #00e5ff; border-radius: 4px;
                color: #00e5ff; background: transparent; font-weight: bold;
            }
            QPushButton:hover { background-color: #00e5ff; color: #050810; }
            QFrame#panel {
                background-color: #0d1120;
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 6px;
            }
        """)

        self.emotion_stats = {label: 0 for label in EMOTION_LABELS}
        self.signals = Signals()
        self.signals.frame_ready.connect(self.update_frame)
        self.fps_count = 0
        self.frame_total = 0
        self._build_ui()
        
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._tick_fps)
        self.fps_timer.start(1000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)
        
        left = QVBoxLayout()
        self.video_label = QLabel("📷 System Ready\nClick Start Detection")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#0d1120; border:1px solid rgba(255,255,255,0.1); border-radius:8px;")
        left.addWidget(self.video_label)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶  START ANALYTICS")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  STOP")
        self.stop_btn.clicked.connect(self.stop)
        btns.addWidget(self.start_btn); btns.addWidget(self.stop_btn); btns.addStretch()
        left.addLayout(btns)
        root.addLayout(left, 3)

        right = QVBoxLayout()
        stats_frame = QFrame(); stats_frame.setObjectName("panel")
        sl = QVBoxLayout(stats_frame)
        grid = QGridLayout()
        
        self.stat_fps, self.val_fps = self._stat("—", "Live FPS")
        self.stat_people, self.val_people = self._stat("0", "People Count")
        self.stat_frames, self.val_frames = self._stat("0", "Total Frames")
        
        grid.addWidget(self.stat_fps, 0, 0)
        grid.addWidget(self.stat_people, 0, 1)
        grid.addWidget(self.stat_frames, 1, 0, 1, 2)
        
        sl.addLayout(grid)
        right.addWidget(stats_frame)

        det_frame = QFrame(); det_frame.setObjectName("panel")
        det_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.det_lbl = QLabel("Awaiting Feed...")
        self.det_lbl.setWordWrap(True)
        self.det_lbl.setAlignment(Qt.AlignTop)
        
        dl = QVBoxLayout(det_frame)
        dl.addWidget(self.det_lbl)
        right.addWidget(det_frame)
        
        root.addLayout(right, 1)

    def _stat(self, val, label):
        f = QFrame(); l = QVBoxLayout(f)
        v = QLabel(val); v.setStyleSheet("color:#00e5ff; font-size:26px; font-weight:bold;")
        lb = QLabel(label.upper()); lb.setStyleSheet("color:#5a6070; font-size:10px; font-weight: bold;")
        l.addWidget(v); l.addWidget(lb)
        return f, v

    def start(self):
        self.app = DetectionApp(self.signals)
        self.app.start()
        self.start_btn.setEnabled(False)

    def stop(self):
        if hasattr(self, 'app'): self.app.stop()
        self.start_btn.setEnabled(True)

    def update_frame(self, data):
        display, faces = data
        self.fps_count += 1; self.frame_total += 1
        self.val_frames.setText(f"{self.frame_total:,}")
        self.val_people.setText(str(len(faces))) 
        
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))

        text = "<b style='font-size:14px; color:#ffffff;'>CURRENT DETECTIONS</b><br>"
        if not faces: text += "<i style='color:#5a6070;'>Scanning...</i>"
        
        for i, face in enumerate(faces):
            emo = face["emotion"]
            self.emotion_stats[emo] += 1 
            color = EMOTION_HEX.get(emo, "#00e5ff")
            text += f"<span style='color:{color}'><b>P#{i+1}:</b> {face['gender']} | {emo}</span><br>"
        
        text += "<br><hr style='background-color:rgba(255,255,255,0.1); border:none; height:1px;'><br>"
        text += "<b style='font-size:14px; color:#ffffff;'>MOOD TRENDS</b><br>"
        total_dets = sum(self.emotion_stats.values())
        if total_dets > 0:
            for emo in EMOTION_LABELS:
                perc = (self.emotion_stats[emo] / total_dets) * 100
                if perc > 0:
                    text += f"<span style='color:#8899aa'>{emo}:</span> <span style='color:#e8eaf0'>{perc:.1f}%</span><br>"

        self.det_lbl.setText(text)
        self.det_lbl.setTextFormat(Qt.RichText)

    def _tick_fps(self):
        self.val_fps.setText(str(self.fps_count))
        self.fps_count = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())