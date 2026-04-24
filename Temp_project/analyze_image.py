import cv2
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace
import sys

EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

emotion_model = load_model("models/emotion_model.h5")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_image(image_path):
    frame = cv2.imread(image_path)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

    print(f"\nFound {len(faces)} face(s) in {image_path}\n")

    for i, (x, y, w, h) in enumerate(faces):
        face_gray = gray[y:y+h, x:x+w]
        face_bgr  = frame[y:y+h, x:x+w]

        # Emotion
        face_in = cv2.resize(face_gray, (48,48)).astype('float32') / 255.0
        face_in = np.expand_dims(face_in, axis=(0,-1))
        preds = emotion_model.predict(face_in, verbose=0)[0]
        emotion = EMOTION_LABELS[np.argmax(preds)]

        # Gender + Age via DeepFace
        try:
            result = DeepFace.analyze(face_bgr, actions=['gender','age'],
                                      enforce_detection=False, silent=True)
            if isinstance(result, list): result = result[0]
            gender = result['dominant_gender']
            age    = result['age']
        except:
            gender, age = "Unknown", "?"

        print(f"  Face {i+1}: Emotion={emotion} | Gender={gender} | Age≈{age}")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{emotion} | {gender} | {age}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out_path = "output_" + image_path.split("/")[-1]
    cv2.imwrite(out_path, frame)
    print(f"\nSaved annotated image → {out_path}")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    analyze_image(img)