import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize keyboard controller for global key simulation
keyboard = Controller()

# Function to determine gesture based on hand landmarks
def get_gesture(landmarks):
    # Landmark indices
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Check if fingers are extended or curled
    def is_finger_extended(tip, mcp, wrist):
        return tip.y < mcp.y and tip.y < wrist.y  # Assuming hand is upright

    index_extended = is_finger_extended(index_tip, index_mcp, wrist)
    middle_extended = is_finger_extended(middle_tip, middle_mcp, wrist)
    ring_extended = is_finger_extended(ring_tip, ring_mcp, wrist)
    pinky_extended = is_finger_extended(pinky_tip, pinky_mcp, wrist)
    thumb_extended = thumb_tip.x > wrist.x  # Rough check for thumb

    # Fist: All fingers curled (not extended) - Removed from key map, but kept for detection if needed
    if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "fist"

    # Open palm: All fingers extended
    if index_extended and middle_extended and ring_extended and pinky_extended:
        return "open_palm"

    # Pinky only: Only pinky extended, others curled
    if pinky_extended and not index_extended and not middle_extended and not ring_extended:
        return "pinky_only"

    # Pointing gestures: Only index extended, others curled
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        # Determine direction based on hand orientation (simplified)
        # If index tip is to the left of wrist, pointing left; else right
        if index_tip.x < wrist.x:
            return "point_left"
        else:
            return "point_right"

    return None

# Main function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    current_keys = set()  # Track currently pressed keys

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = hands.process(rgb_frame)

        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get gesture
                landmarks = hand_landmarks.landmark
                gesture = get_gesture(landmarks)

        # Map gesture to keys (list of keys to press)
        key_map = {
            "open_palm": ["w"],
            "pinky_only": ["s"],
            "point_left": ["w", "a"],
            "point_right": ["w", "d"]
        }

        target_keys = set(key_map.get(gesture, []))

        # Handle key presses and releases
        keys_to_release = current_keys - target_keys
        keys_to_press = target_keys - current_keys

        for key in keys_to_release:
            keyboard.release(key)
        for key in keys_to_press:
            keyboard.press(key)

        current_keys = target_keys

        # Display gesture on frame
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Hand Gesture Control", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup: Release all pressed keys
    for key in current_keys:
        keyboard.release(key)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
