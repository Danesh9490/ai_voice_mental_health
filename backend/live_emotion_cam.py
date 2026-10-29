import cv2
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from mental_state import analyze_mental_state
from collections import deque
from reportlab.pdfgen import canvas

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load emotion model
model = load_model("emotion_cnn.h5")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# Store last predictions for smoothing
emotion_buffer = deque(maxlen=10)

session_start = datetime.datetime.now()

cap = cv2.VideoCapture(0)

print("Live Face Tracking Started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60,60)
    )

    if len(faces) > 0:

        # take biggest face only
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (96,96))
        face = face / 255.0
        face = np.reshape(face, (1,96,96,3))

        prediction = model.predict(face, verbose=0)

        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        emotion_buffer.append(emotion)

        # Most common emotion in buffer
        final_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        depression_score = 1 if final_emotion in ["sad","fear"] else 0

        mental_state = analyze_mental_state(final_emotion, depression_score)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            frame,
            f"Emotion: {final_emotion}",
            (x,y-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"State: {mental_state}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

    cv2.imshow("AI Mental Health Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# Save final report
pdf = canvas.Canvas("session_report.pdf")

pdf.drawString(200,800,"AI Mental Health Session Report")
pdf.drawString(50,760,f"Session Time: {session_start}")

if len(emotion_buffer) > 0:

    final_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

    depression_score = 1 if final_emotion in ["sad","fear"] else 0

    final_state = analyze_mental_state(final_emotion, depression_score)

    pdf.drawString(50,720,f"Detected Emotion: {final_emotion}")
    pdf.drawString(50,700,f"Mental State: {final_state}")

pdf.save()

print("Report saved: session_report.pdf")