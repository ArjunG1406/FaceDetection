import cv2
import numpy as np

def show_intermediate_steps():
    # Use the same detector as your main app
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    print("🚀 Intermediate Dashboard Started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Standard Gray Conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Create a placeholder for the right-side panel if no face is detected
        # We'll make it 400x200 to fit two 200x200 blocks
        side_panel = np.zeros((400, 200, 3), dtype="uint8")

        for (x, y, w, h) in faces:
            # 2. Extract Raw Crop & Resize for Display
            face_crop = frame[y:y+h, x:x+w]
            face_crop_res = cv2.resize(face_crop, (200, 200))

            # 3. Model Input View (The 48x48 Grayscale)
            # This is exactly what the CNN sees!
            input_48 = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            
            # Convert back to BGR so we can stack it with color images
            input_view = cv2.cvtColor(input_48, cv2.COLOR_GRAY2BGR)
            input_view = cv2.resize(input_view, (200, 200), interpolation=cv2.INTER_NEAREST)

            # Stack them vertically for the side panel
            side_panel = np.vstack((face_crop_res, input_view))

            # Draw green box on the main feed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break # Only process the first face to keep the debug view clean

        # Resize main frame to match the height of the side panel (400px)
        main_res = cv2.resize(frame, (533, 400)) 
        
        # Combine everything into ONE window
        combined_dashboard = np.hstack((main_res, side_panel))

        # Add some labels so the guide knows what they are looking at
        cv2.putText(combined_dashboard, "1. RAW CROP", (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(combined_dashboard, "2. CNN INPUT", (540, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Internal Processing Pipeline", combined_dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_intermediate_steps()