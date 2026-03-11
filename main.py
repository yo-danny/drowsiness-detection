import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np

# Modelo dos marcos faciais
model_path = "face_landmarker.task"


def init_detector():
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

    detector = FaceLandmarker.create_from_options(options)

    return detector


cv2.namedWindow("MediaPipe Face Landmarker")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
    detector = init_detector()
else:
    rval = False

while rval:

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    result = detector.detect_for_video(image, timestamp)

    if result.face_landmarks:

        for landmark in result.face_landmarks:

            # drawing_utils.draw_landmarks(
            #     frame,
            #     landmark,
            #     vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            #     None,
            #     drawing_styles.get_default_face_mesh_tesselation_style(),
            # )

            face_landmarks_list = result.face_landmarks[0]

            chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
            chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]

            all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

            all_chosen_points = [face_landmarks_list[i] for i in all_chosen_idxs]

            frame_heigth, frame_width, _ = frame.shape

            for point in all_chosen_points:
                x, y = int(point.x * frame_width), int(point.y * frame_heigth)

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("MediaPipe Face Landmarker", frame)
    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:
        break
    if cv2.getWindowProperty("MediaPipe Face Landmarker", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyWindow("MediaPipe Face Landmarker")
vc.release()

# # Pontos escolhidos : P1, P2, P3, P4, P5, P6

# image = cv2.imread(r"test-open-eyes.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_GR2RGB)
# image = np.ascontiguousarray(image)
# imgH, imgW, _ = image.shape

# plt.imshow(image)
