import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from scipy import spatial
import cvzone
import time

# pip install opencv-python mediapipe numpy pyautogui scipy cvzone
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

def get_landmark(landmarks, landmark_id):
    """Helper to get landmark coordinates"""
    return landmarks[landmark_id]

def detect_gesture(landmarks):
    """Detect specific gestures based on landmark positions"""
    
    # Get all necessary landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if other fingers are folded
    other_fingers_folded = (
        index_tip.y > index_pip.y and
        middle_tip.y > middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y
    )
    
    # 👍 Thumbs Up → W
    thumb_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y)
    if thumb_up and other_fingers_folded:
        return "thumbs_up"
    
    # 👎 Thumbs Down → S
    thumb_down = (thumb_tip.y > thumb_ip.y and thumb_tip.y > thumb_mcp.y)
    if thumb_down and other_fingers_folded:
        return "thumbs_down"
    
    # ✌️ Peace Sign → A
    peace = (
        index_tip.y < index_pip.y and
        middle_tip.y < middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y
    )
    if peace:
        return "peace"
    
    # ✊ Fist → D
    fist = (
        thumb_tip.y > thumb_mcp.y and
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )
    if fist:
        return "fist"
    
    return None

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            gesture = detect_gesture(landmarks)

            if gesture == "thumbs_up":
                pyautogui.press('w')
                cv2.putText(frame, "W - Thumbs Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture == "thumbs_down":
                pyautogui.press('s')
                cv2.putText(frame, "S - Thumbs Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture == "peace":
                pyautogui.press('a')
                cv2.putText(frame, "A - Peace", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture == "fist":
                pyautogui.press('d')
                cv2.putText(frame, "D - Fist", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Control", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
