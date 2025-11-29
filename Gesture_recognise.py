import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time


# Strong TTS (resets engine every time)

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)      
    engine.setProperty('volume', 1.0)     
    engine.say(text)
    engine.runAndWait()
    engine.stop()


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

last_char = None
cooldown = 0.3

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'C', 4: 'S', 5: '7'}

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        x_ = []; y_ = []; data_aux = []

        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        for lm in hand.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) != 42:
            continue

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])[0]
        proba = model.predict_proba([np.asarray(data_aux)])[0]

        predicted_character = labels_dict[int(prediction)]
        confidence = proba[int(prediction)]

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
        cv2.putText(frame, predicted_character, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
        cv2.putText(frame, f"{confidence*100:.2f}%", (x1, y2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        # SPEAK — no queue,no delay skip
        if confidence > 0.6:
            if predicted_character != last_char:
                threading.Thread(target=speak, args=(predicted_character,), daemon=True).start()
                last_char = predicted_character

    small_frame = cv2.resize(frame, (200, 200))  # any size you want
    cv2.imshow("frame", small_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
