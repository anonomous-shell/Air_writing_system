import cv2
import numpy as np
import pytesseract
import pyttsx3
import time
from src.hand_tracker import HandTracker
from src.drawing_utils import AirCanvas

# 👉 Tesseract path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔊 Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak_text(text):
    if text:
        engine.say(text)
        engine.runAndWait()


def fingers_up(lm_list):
    if len(lm_list) == 0:
        return []

    fingers = []
    fingers.append(lm_list[8][2] < lm_list[6][2])   # Index
    fingers.append(lm_list[12][2] < lm_list[10][2]) # Middle
    fingers.append(lm_list[16][2] < lm_list[14][2]) # Ring
    fingers.append(lm_list[20][2] < lm_list[18][2]) # Pinky

    return fingers


# 🔥 OCR FUNCTION (clean white canvas)
def recognize_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        thresh,
        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    )

    return text.strip()


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    canvas = AirCanvas()

    detected_text = ""
    last_spoken_time = 0   # ⏱️ time control
    ocr_canvas = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # 🔥 create white OCR canvas
        if ocr_canvas is None:
            ocr_canvas = 255 * np.ones_like(frame)

        canvas.initialize_canvas(frame)

        tracker.find_hands(frame)
        lm_list = tracker.get_landmarks(frame)
        tracker.draw_hands(frame)

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1], lm_list[8][2]
            fingers = fingers_up(lm_list)

            # ✊ Clear
            if fingers == [False, False, False, False]:
                canvas.canvas = frame.copy() * 0
                ocr_canvas = 255 * np.ones_like(frame)
                detected_text = ""
                last_spoken_time = 0

            # ✌️ Move
            elif fingers[0] and fingers[1] and not fingers[2]:
                canvas.reset_position()

            # ✍️ Draw
            elif fingers[0] and not fingers[1]:
                canvas.draw(frame, x1, y1)

                # draw on OCR canvas (black ink)
                cv2.circle(ocr_canvas, (x1, y1), 10, (0, 0, 0), -1)

            # 🤟 OCR + Repeat Voice with delay
            elif fingers[0] and fingers[1] and fingers[2]:
                new_text = recognize_text(ocr_canvas)

                current_time = time.time()

                if new_text != "" and (current_time - last_spoken_time > 2):
                    detected_text = new_text
                    print("Detected Text:", detected_text)

                    speak_text(detected_text)  # 🔊 speak

                    last_spoken_time = current_time

        # merge drawing
        frame = canvas.merge_canvas(frame)

        # show detected text
        cv2.putText(frame, detected_text, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        cv2.imshow("Air Writing + OCR + Voice", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
