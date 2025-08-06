import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

# Emoji map
emoji_map = {
    "Happy": "😊", "Sad": "😢", "Angry": "😠", "Surprise": "😲",
    "Fear": "😨", "Disgust": "🤢", "Neutral": "😐"
}

print("🔄 Loading emotion recognition model...")
try:
    model = load_model("models/emotion_model.hdf5", compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_counter = defaultdict(int)

print("🔄 Loading Haar Cascade for face detection...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("✅ Haar Cascade loaded!")

print("🎥 Opening webcam...")
cap = cv2.VideoCapture(0)
print("📷 Webcam opened:", cap.isOpened())

# Setup for video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_emotion.avi', fourcc, 20.0, (640, 480))

if not cap.isOpened():
    print("❌ Webcam failed to open.")
else:
    print("🎬 Starting real-time emotion detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (64, 64))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, 64, 64, 1))

            prediction = model.predict(roi_reshaped, verbose=0)
            label = emotion_labels[np.argmax(prediction)]
            emoji = emoji_map[label]

            # Show label + emoji
            cv2.putText(frame, f"{label} {emoji}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Count & log
            emotion_counter[label] += 1
            with open("emotion_log.txt", "a") as log:
                log.write(f"{timestamp} - {label}\n")

        # Show frame
        cv2.imshow("Real-Time Emotion Recognition - Press 'q' to Quit | 's' to Screenshot", frame)
        out.write(frame)  # Save video frame

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("captured_face.png", frame)
            print("📸 Screenshot saved as 'captured_face.png'")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n📊 Emotion Detection Summary:")
    for emo, count in emotion_counter.items():
        print(f"{emo}: {count}")

    # Pie Chart
    if emotion_counter:
        plt.figure(figsize=(6, 6))
        labels = list(emotion_counter.keys())
        sizes = list(emotion_counter.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Emotion Distribution")
        plt.axis('equal')
        plt.show()

        print("🏁 Program ended. Thanks for using Face Emotion Detector! ❤️")

        #✅ Current Features in Your Code
#Here’s what your code already does:

#🔢	Feature	Description
#1️⃣	Webcam Access	Opens webcam and reads real-time video
#2️⃣	Face Detection	Uses Haar Cascade to detect faces
#3️⃣	Emotion Recognition (CNN)	Classifies face emotions using pre-trained model
#4️⃣	Text Label on Screen	Shows emotion label (ex: Happy, Sad) in live video

#5️⃣	Logs Emotions with Time 🕒	Appends results to emotion_log.txt with timestamps
#6️⃣	Take Screenshot (s key) 📸	Saves a screenshot of the current frame
#7️⃣	Counts Emotion Summary	Shows how many times each emotion was detected
#8️⃣	Emoji Display 😀	Shows emoji next to emotion label
#9️⃣	Live Video Save 🎥	Saves session as output_emotion.avi
#🔟	Pie Chart of Emotion Summary 📊	Displays pie chart after quitting the app 

# 1.cd "E:\projects 2025\face reco\Face_Emotion_Recognition"
#2.dir scripts

    # 3. how to run ..use this command...
    