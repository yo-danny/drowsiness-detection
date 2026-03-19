import cv2
import mediapipe as mp
import chime
import time

# Facial landmark model
model_path = "face_landmarker.task"


class EARCalculator:
    @staticmethod
    def distance(p1, p2):
        # Return the distance between two points in a cartesian plan
        dist = sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5
        return dist

    @staticmethod
    def compute(points):
        P2_P6 = EARCalculator.distance(points[1], points[5])
        P3_P5 = EARCalculator.distance(points[2], points[4])
        P1_P4 = EARCalculator.distance(points[0], points[3])
        return (P2_P6 + P3_P5) / (2 * P1_P4)


class DrowsinessDetector:
    EAR_LIMIT = 0.15
    DROWSY_SECONDS = 1.0

    def __init__(self, model_path):
        self.detector = self.init_detector(model_path)
        self.eyes_closed_start = None
        self.is_drowsy = False

        # Chosen points: P1,  P2,  P3,  P4,  P5,  P6
        self.left_eye = [362, 385, 387, 263, 373, 380]
        self.right_eye = [33, 160, 158, 133, 153, 144]

    def init_detector(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Cria a instância de com o modo de vídeo
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )

        return FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = self.detector.detect_for_video(image, timestamp)

        if not result.face_landmarks:
            return frame

        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        coords = [(int(p.x * w), int(p.y * h)) for p in landmarks]

        left_points = [coords[i] for i in self.left_eye]
        right_points = [coords[i] for i in self.right_eye]

        left_ear = EARCalculator.compute(left_points)
        right_ear = EARCalculator.compute(right_points)
        ear = (left_ear + right_ear) / 2

        self._check_drowsiness(ear)

        color = (0, 255, 0) if not self.is_drowsy else (0, 0, 255)

        for x, y in left_points + right_points:
            cv2.circle(frame, (x, y), 2, color, -1)

        cv2.putText(
            frame,
            f"EAR: {round(ear, 2)}",
            (1, 24),
            cv2.FONT_HERSHEY_COMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        return frame

    def _check_drowsiness(self, ear):
        now = time.time()

        if ear < self.EAR_LIMIT:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = now

            if now - self.eyes_closed_start >= self.DROWSY_SECONDS:
                if not self.is_drowsy:
                    chime.warning()
                self.is_drowsy = True
        else:
            self.eyes_closed_start = None
            self.is_drowsy = False


def main():
    cap = cv2.VideoCapture(0)
    detector = DrowsinessDetector("face_landmarker.task")

    if not cap.isOpened():
        print("Erro ao acessar a webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.process_frame(frame)

        cv2.imshow("Drowsiness Detector", frame)

        key = cv2.waitKey(20)
        if key == 27:
            break
        if cv2.getWindowProperty("Drowsiness Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
