import cv2
import numpy as np
import serial
import time
import gc  # Untuk garbage collection

# Inisialisasi background subtractor dengan parameter yang lebih hemat memory
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Buka kamera dengan resolusi 640x480
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 15)  # Menurunkan FPS untuk mengurangi penggunaan CPU dan memory

# Inisialisasi Serial ke Arduino
try:
    arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)
    print("Koneksi ke Arduino berhasil")
except:
    print("Gagal koneksi ke Arduino")
    arduino = None

last_sent = 0
send_interval = 0.05  # 100 ms, mengurangi frekuensi pengiriman data ke Arduino
last_angle = 0  # Untuk tracking angle terakhir yang dikirim
angle_threshold = 3  # Hanya kirim jika perubahan angle lebih dari threshold

# Counter untuk pembersihan memory
frame_count = 0
cleanup_interval = 30  # Lakukan pembersihan setiap 30 frame

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame untuk menghemat memory jika diperlukan
        # frame = cv2.resize(frame, (320, 240))
        
        # Convert to grayscale untuk mengurangi penggunaan memory
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        
        # Filter noise dengan kernel yang lebih kecil
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # Hanya 1 iterasi dilate
        fgmask = cv2.dilate(fgmask, None, iterations=1)
        
        # Hanya proses contour pada tiap 2 frame untuk menghemat CPU
        process_contours = (frame_count % 2 == 0)
        
        if process_contours:
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contour berdasarkan area
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1500]
            
            # Hanya proses jika ada contour yang signifikan
            if significant_contours:
                rects = [cv2.boundingRect(cnt) for cnt in significant_contours]
                
                # Cari bounding box global
                x_min = min([x for x, y, w, h in rects])
                y_min = min([y for x, y, w, h in rects])
                x_max = max([x + w for x, y, w, h in rects])
                y_max = max([y + h for x, y, w, h in rects])

                # Gambar bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Hitung posisi tengah objek
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Mapping posisi X ke sudut servo
                angle = int(np.interp(center_x, [0, FRAME_WIDTH], [120, 30]))
                cv2.putText(frame, f"Angle: {angle}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Kirim ke Arduino hanya jika angle berubah signifikan
                now = time.time()
                if arduino and now - last_sent >= send_interval and abs(angle - last_angle) >= angle_threshold:
                    arduino.write(f"{angle}\n".encode())
                    arduino.flush()
                    last_sent = now
                    last_angle = angle
        
        # Tampilkan hasil
        cv2.imshow("Frame", frame)
        cv2.imshow("Foreground Mask", fgmask)
        
        # Increment frame counter
        frame_count += 1
        
        # Periodic memory cleanup
        if frame_count % cleanup_interval == 0:
            fgbg.apply(np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8))
            gc.collect()  # Paksa garbage collection
        
        # Cap frame rate to reduce CPU usage
        key = cv2.waitKey(30) & 0xFF  # 30ms delay (~33 FPS max)
        if key == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    if arduino:
        arduino.close()
    cv2.destroyAllWindows()
    gc.collect()  # Final garbage collection
