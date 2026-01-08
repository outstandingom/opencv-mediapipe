import cv2
import numpy as np
import mediapipe as mp
"""pip install opencv-python mediapipe numpy
"""
# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 2. Initialize Webcam
cap = cv2.VideoCapture(0)

# Variables to keep track of drawing state
prev_x, prev_y = 0, 0
canvas = None # We will initialize this once we know the frame size

print("--- AIR CANVAS STARTED ---")
print("Draw with your INDEX FINGER.")
print("Press 'c' to CLEAR canvas.")
print("Press 'q' to QUIT.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a "mirror" effect (more natural for drawing)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Initialize canvas if it doesn't exist
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # 3. Hand Tracking Logic
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get coordinates of Index Finger Tip (Landmark ID 8)
            lm = hand_landmarks.landmark[8] 
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Visual Indicator: Draw a circle on the finger tip
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Drawing Logic: Connect previous point to current point
            # We check if prev_x is 0 to avoid drawing a line from the corner on first detection
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Draw a line on the canvas mask
            cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 255), 8)
            
            # Update previous coordinates
            prev_x, prev_y = cx, cy
            
            # (Optional) Draw hand skeleton on the main frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # If hand is lost, reset previous coordinates so we don't draw a generic line when it returns
        prev_x, prev_y = 0, 0

    # 4. Merge the Canvas with the Video Feed
    # We create a gray version of the canvas to create a mask
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Isolate the foreground (the drawing) and the background (the video)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    # Display the result
    cv2.imshow('Air Canvas - Python', frame)

    # Key controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8) # Clear canvas

cap.release()
cv2.destroyAllWindows()
