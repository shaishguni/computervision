## Hand Gesture Keyboard Controller

Control any app that understands the `W`, `A`, `S`, `D` keys with simple hand gestures tracked by your webcam. The script in `main.py` uses OpenCV and MediaPipe Hands to detect thumb, index, middle, ring, and pinky positions, then sends virtual key presses through `pyautogui`.

### Features
- Real-time hand-tracking with MediaPipe.
- Four mapped gestures (👍, 👎, ✌️, ✊) that trigger WASD key presses.
- Visual overlay showing detected hand landmarks and the active gesture.

### Requirements
- Python 3.9+ (tested on macOS, should work on Windows/Linux with minor tweaks).
- Webcam with stable lighting.
- Dependencies:
	```bash
	pip install opencv-python mediapipe numpy pyautogui scipy cvzone
	```
	> `pyautogui` may require Accessibility permissions on macOS.

### Getting Started
1. Clone or download this repo.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies using the command above.
4. Run the script:
	 ```bash
	 python main.py
	 ```
5. Allow camera access when prompted and position your hand so that the camera can see the full palm.

### Gesture Mapping
| Gesture | Action | Key |
| --- | --- | --- |
| 👍 Thumbs Up (other fingers folded) | Move forward | `W` |
| 👎 Thumbs Down | Move backward | `S` |
| ✌️ Peace Sign | Move left | `A` |
| ✊ Fist | Move right | `D` |

### Tips
- Keep the background uncluttered and ensure adequate lighting for better detection.
- If gestures misfire, adjust hand distance from the camera or tweak the confidence thresholds in `hands = mp_hands.Hands(...)`.
- To map gestures to other keys, update the `pyautogui.press(...)` calls inside the gesture conditionals.

### Troubleshooting
- **Window immediately closes**: Ensure your webcam is free and not used by another app.
- **High latency**: Reduce frame size or lower the number of tracked hands to keep processing light.
- **No key presses on macOS**: Grant Accessibility permissions to the Python interpreter in *System Settings → Privacy & Security → Accessibility*.

Enjoy experimenting with hands-free controls!
