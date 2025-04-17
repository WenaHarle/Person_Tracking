import cv2
import numpy as np

# Inisialisasi background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=10, detectShadows=False)

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Background subtraction
    fgmask = fgbg.apply(frame)
    
    # Filter noise dengan operasi morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Ukuran kernel diperbesar
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Tambahkan dilasi agar objek lebih menyatu
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    
    # Temukan kontur
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gabungkan bounding box yang berdekatan
    rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    if rects:
        x_min = min([x for x, y, w, h in rects])
        y_min = min([y for x, y, w, h in rects])
        x_max = max([x + w for x, y, w, h in rects])
        y_max = max([y + h for x, y, w, h in rects])
        
        # Gambar bounding box menyeluruh
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Tampilkan hasil
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground Mask", fgmask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
