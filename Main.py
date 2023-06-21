import cv2
import numpy as np

# Velg video
cap = cv2.VideoCapture("edit1.mp4")

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Finn ønsket høyde / bredde for video
display_width = 1200  # Endre denne for størrelse
display_height = int(original_height * (display_width / original_width))

# Find all dark objects on a white background
object_detector = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=1000, detectShadows=True)


while True:
    # Les frame
    ret, frame = cap.read()

    # hvis frame ikke er firkant så avslutt
    if not ret:
        break

    # Ny video til ønsket størrelse
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Endre frame til svart-hvitt
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Finn mørke objekter på hvit bagrunn og mask
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Lag kontorer slik at vi ser maske i preview
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process the contours and hierarchy
    for i, cnt in enumerate(contours):
        # Fjern noise i video
        area = cv2.contourArea(cnt)

        # Må til pga colorbanding
        perimeter = cv2.arcLength(cnt, True)

        # Fiks for fiks for colorbanding
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # Ignorer småområder, slå disse sammen
            if area > 500 and hierarchy[0, i, 3] == -1 and circularity > 0.05:
                x, y, w, h = cv2.boundingRect(cnt)
                # Tegn grønn rektangel
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Vis maske og frame
    cv2.imshow("frame", resized_frame)
    cv2.imshow("mask", mask)

    # Trykk X for og lukke preview
    key = cv2.waitKey(30)
    if key == ord('x') or key == ord('X'):
        break

cap.release()
cv2.destroyAllWindows()
