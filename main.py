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

# Map each gesture to the keys that should be held simultaneously.
GESTURE_KEY_MAP = {
    "thumbs_up": ("w",),
    "thumbs_down": ("s",),
    "peace": ("w", "a"),
    "open_hand": ("d",),
}

GESTURE_LABEL = {
    "thumbs_up": "W - Thumbs Up",
    "thumbs_down": "S - Thumbs Down",
    "peace": "W + A - Peace",
    "open_hand": "D - Open Hand",
}

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
    
    # ✌️ Peace Sign → W + A
    peace = (
        index_tip.y < index_pip.y and
        middle_tip.y < middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y
    )
    if peace:
        return "peace"
    
    # ✋ Open Hand → D
    open_hand = (
        index_tip.y < index_pip.y and
        middle_tip.y < middle_pip.y and
        ring_tip.y < ring_pip.y and
        pinky_tip.y < pinky_pip.y
    )
    if open_hand:
        return "open_hand"
    
    return None

# UI: on-screen Exit button
BUTTON = (10, 10, 120, 40)  # x, y, w, h
state = {"exit": False}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = BUTTON
        if bx <= x <= bx + bw and by <= y <= by + bh:
            state["exit"] = True

cv2.namedWindow("Hand Gesture Control")
cv2.setMouseCallback("Hand Gesture Control", on_mouse)

active_keys = set()

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

            desired_keys = set(GESTURE_KEY_MAP.get(gesture, ()))

            # Release keys that are no longer needed.
            for key in active_keys - desired_keys:
                pyautogui.keyUp(key)

            # Press keys newly required by the current gesture.
            for key in desired_keys - active_keys:
                pyautogui.keyDown(key)

            active_keys = desired_keys

            label = GESTURE_LABEL.get(gesture)
            if label:
                cv2.putText(frame, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not result.multi_hand_landmarks and active_keys:
        for key in list(active_keys):
            pyautogui.keyUp(key)
        active_keys.clear()

    # Draw Exit button
    bx, by, bw, bh = BUTTON
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), -1)
    cv2.putText(frame, "Exit", (bx + 25, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Control", frame)

    if state["exit"]:
        break
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

camera.release()
if active_keys:
    for key in list(active_keys):
        pyautogui.keyUp(key)
    active_keys.clear()

cv2.destroyAllWindows()
