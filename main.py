import cv2
from src.detector import SleepDetector

def draw_text_with_background(frame, text, position, font, font_scale, text_color, bg_color, thickness):
    """Status text"""
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    padding = 5
    cv2.rectangle(frame, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def main():
    detector = SleepDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    print("Mulai deteksi. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak bisa membaca frame dari kamera.")
            break

        # Proses untuk face dan pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = True
        face_results = detector.face_mesh.process(rgb_frame)
        pose_results = detector.pose.process(rgb_frame)

        face_status = detector.detect_sleep_and_mouth(frame)
        movement_status = detector.detect_movement(frame)
        pacifier_status = detector.detect_pacifier(frame) 

        # Gambar landmark
        detector.draw_landmarks(frame, face_results, pose_results)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  
        text_color = (0, 255, 0)
        bg_color = (0, 0, 0)
        thickness = 2

        status_lines = face_status.split('\n') + [f"Pergerakan: {movement_status}", pacifier_status]  # Tambah pacifier status
        y_offset = 30
        for line in status_lines:
            draw_text_with_background(frame, line, (10, y_offset), font, font_scale, text_color, bg_color, thickness)
            y_offset += 25

        cv2.imshow("Deteksi Tidur/Bangun, Mulut & Pergerakan - MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()