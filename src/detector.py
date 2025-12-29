import cv2
import numpy as np
import mediapipe as mp
from src.utils import preprocess_frame, eye_aspect_ratio

def mouth_aspect_ratio_mp(mouth_points, landmarks, h, w):
    """Hitung MAR menggunakan landmarks MediaPipe untuk mulut."""
    # MediaPipe face mesh indices for mouth:
    # - Upper lip center: 13
    # - Lower lip center: 14
    # - Left corner: 61
    # - Right corner: 291
    upper_lip = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lower_lip = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left_corner = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right_corner = np.array([landmarks[291].x * w, landmarks[291].y * h])

    vert_dist = np.linalg.norm(upper_lip - lower_lip)
    horiz_dist = np.linalg.norm(left_corner - right_corner)
    mar = vert_dist / (horiz_dist + 1e-6)
    return mar

class SleepDetector:
    def __init__(self):
        """Inisialisasi MediaPipe untuk face mesh dan pose."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
            max_num_faces=1,
            refine_landmarks=True  # Untuk akurasi mata/mulut
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        )
        self.prev_landmarks = None

    def enhance_frame(self, frame):
        """Tingkatkan kontras frame untuk deteksi lebih baik pada jarak jauh."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or invalid")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        return enhanced_frame

    def detect(self, frame):
        """Deteksi semua status (mata, mulut, pergerakan, empeng, rollover) dan kembalikan sebagai dict."""
        if frame is None or frame.size == 0:
            return {
                "error": "Frame is empty or invalid",
                "eye_status": "Tidak terdeteksi",
                "mouth_status": "Tidak terdeteksi",
                "movement_status": "Tidak terdeteksi",
                "pacifier_status": "Tidak terdeteksi",
                "rollover_status": "Tidak terdeteksi"
            }

        h, w, _ = frame.shape
        enhanced_frame = self.enhance_frame(frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = True
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        results = {
            "eye_status": "Tidak terdeteksi",
            "mouth_status": "Tidak terdeteksi",
            "movement_status": "Tidak terdeteksi",
            "pacifier_status": "Tidak terdeteksi",
            "rollover_status": "Tidak terdeteksi"
        }

        # Face detection (eyes and mouth)
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # Eye detection
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices])
            right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices])
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            if ear < 0.2:
                results["eye_status"] = f"Mata tertutup [EAR: {ear:.2f}]"
            elif ear > 0.25:
                results["eye_status"] = f"Mata terbuka [EAR: {ear:.2f}]"
            else:
                results["eye_status"] = f"Mata setengah terbuka [EAR: {ear:.2f}]"

            # Mouth detection
            try:
                mar = mouth_aspect_ratio_mp(None, landmarks, h, w)
                results["mouth_status"] = f"Mulut terbuka [MAR: {mar:.2f}]" if mar > 0.4 else f"Mulut tertutup [MAR: {mar:.2f}]"
            except Exception as e:
                results["mouth_status"] = f"Error MAR: {str(e)}"

            # Pacifier detection
            results["pacifier_status"] = self.detect_pacifier(frame)

        # Movement and rollover detection
        results["movement_status"] = self.detect_movement(frame)
        results["rollover_status"] = self.detect_rollover(frame, pose_results)

        return results

    def detect_movement(self, frame):
        """Deteksi pergerakan dengan threshold sensitif untuk bayi."""
        enhanced_frame = self.enhance_frame(frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = True
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            self.prev_landmarks = None
            return "Tubuh tidak terdeteksi, pergerakan tidak terukur."

        landmarks = results.pose_landmarks.landmark

        selected_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        current_landmarks = {
            idx: (landmarks[idx.value].x, landmarks[idx.value].y, landmarks[idx.value].z)
            for idx in selected_landmarks
        }

        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks.copy()
            return "Tidak ada pergerakan (inisialisasi)"

        total_distance = 0
        for idx in current_landmarks:
            curr = current_landmarks[idx]
            prev = self.prev_landmarks[idx]
            distance = np.sqrt(sum((c - p)**2 for c, p in zip(curr, prev)))
            total_distance += distance

        avg_distance = total_distance / len(current_landmarks)
        threshold = 0.03
        movement_status = "Bergerak!" if avg_distance > threshold else "Tidak bergerak."

        self.prev_landmarks = current_landmarks.copy()
        return movement_status

    def detect_pacifier(self, frame):
        """Deteksi empeng di mulut (perkiraan dengan color thresholding dan contour)."""
        h, w, _ = frame.shape
        enhanced_frame = self.enhance_frame(frame)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = True
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return "Empeng: Tidak terdeteksi (wajah hilang)"

        landmarks = results.multi_face_landmarks[0].landmark

        # Ekstrak ROI mulut berdasarkan landmark (left corner 61, right corner 291, upper 13, lower 14)
        left_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
        right_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
        upper_lip = (int(landmarks[13].x * w), int(landmarks[13].y * h))
        lower_lip = (int(landmarks[14].x * w), int(landmarks[14].y * h))

        # ROI mulut: Kotak sekitar mulut dengan padding
        min_x = max(min(left_corner[0], right_corner[0]) - 10, 0)
        max_x = min(max(left_corner[0], right_corner[0]) + 10, w)
        min_y = max(min(upper_lip[1], lower_lip[1]) - 10, 0)
        max_y = min(max(upper_lip[1], lower_lip[1]) + 10, h)
        mouth_roi = frame[min_y:max_y, min_x:max_x]

        if mouth_roi.size == 0:
            return "Empeng: Tidak terdeteksi (ROI kosong)"

        # Convert ROI ke HSV untuk color thresholding (asumsi empeng putih/cerah)
        hsv_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 0, 150])  # Putih/cerah
        upper_color = np.array([180, 100, 255])
        mask = cv2.inRange(hsv_roi, lower_color, upper_color)

        # Temukan contour di mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Deteksi empeng jika ada contour matching ukuran/bentuk
        pacifier_detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 500:  # Ukuran empeng perkiraan
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), closed=True)
                if len(approx) > 5:  # Bentuk oval/bulat
                    pacifier_detected = True
                    break

        return "Empeng: Dipakai" if pacifier_detected else "Empeng: Tidak pakai"

    def detect_rollover(self, frame, pose_results):
        """Deteksi rollover berdasarkan posisi bahu dan pinggul."""
        if not pose_results.pose_landmarks:
            return "Rollover: Tidak terdeteksi (tubuh hilang)"

        landmarks = pose_results.pose_landmarks.landmark
        h, w, _ = frame.shape

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Konversi ke koordinat piksel
        left_shoulder_pos = (left_shoulder.x * w, left_shoulder.y * h)
        right_shoulder_pos = (right_shoulder.x * w, right_shoulder.y * h)
        left_hip_pos = (left_hip.x * w, left_hip.y * h)
        right_hip_pos = (right_hip.x * w, right_hip.y * h)

        # Hitung perbedaan vertikal (y) untuk bahu dan pinggul
        shoulder_y_diff = abs(left_shoulder_pos[1] - right_shoulder_pos[1])
        hip_y_diff = abs(left_hip_pos[1] - right_hip_pos[1])

        # Threshold untuk deteksi rollover (bayi miring/tengkurap)
        rollover_threshold = 30  # Piksel
        is_rollover = shoulder_y_diff > rollover_threshold or hip_y_diff > rollover_threshold

        return "Rollover: Ya" if is_rollover else "Rollover: Tidak"

    def detect_sleep_and_mouth(self, frame):
        """Deteksi status mata dan mulut untuk backward compatibility."""
        results = self.detect(frame)
        eye_status = results.get('eye_status', 'Tidak terdeteksi')
        mouth_status = results.get('mouth_status', 'Tidak terdeteksi')
        return f"{eye_status}\n{mouth_status}"

    def draw_landmarks(self, frame, face_results, pose_results):
        """Gambar landmarks pada frame untuk debugging."""
        h, w, _ = frame.shape
        
        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)