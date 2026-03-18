import cv2
import mediapipe as mp
import winsound

# Facial landmark model
model_path = "face_landmarker.task"

# State variable
EAR_LIMIT = 0.15


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


def calculate_distance(point_1, point_2):
    # Return the distance between two points in a cartesian plan
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


cv2.namedWindow("MediaPipe Face Landmarker")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
    detector = init_detector()

    # Chosen points:        P1,  P2,  P3,  P4,  P5,  P6
    chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
    chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]

    all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs
else:
    rval = False

while rval:

    is_drown = False

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    result = detector.detect_for_video(image, timestamp)

    if result.face_landmarks:

        for landmark in result.face_landmarks:
            # To draw all the facial landmarks
            # drawing_utils.draw_landmarks(
            #     frame,
            #     landmark,
            #     vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            #     None,
            #     drawing_styles.get_default_face_mesh_tesselation_style(),
            # )

            face_landmarks_list = result.face_landmarks[0]

            all_chosen_points = [face_landmarks_list[i] for i in all_chosen_idxs]

            frame_heigth, frame_width, _ = frame.shape

            coord_points = []

            for point in all_chosen_points:
                x, y = int(point.x * frame_width), int(point.y * frame_heigth)
                coord_points.append([x, y])

                cv2.circle(
                    frame, (x, y), 2, ((0, 255, 0) if not is_drown else (0, 0, 255)), -1
                )

            # Left eye EAR points
            P2_P6 = calculate_distance(coord_points[1], coord_points[5])
            P3_P5 = calculate_distance(coord_points[2], coord_points[4])
            P1_P4 = calculate_distance(coord_points[0], coord_points[3])

            left_eye_ear = (P2_P6 + P3_P5) / (2 * P1_P4)

            # Right eye EAR points
            P2_P6 = calculate_distance(coord_points[7], coord_points[11])
            P3_P5 = calculate_distance(coord_points[8], coord_points[10])
            P1_P4 = calculate_distance(coord_points[6], coord_points[9])

            right_eye_ear = (P2_P6 + P3_P5) / (2 * P1_P4)

            EAR = (left_eye_ear + right_eye_ear) / 2

            cv2.putText(
                frame,
                f"EAR: {round(EAR, 2)}",
                (1, 24),
                cv2.FONT_HERSHEY_COMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            if EAR < EAR_LIMIT:
                is_drown = True
                winsound.Beep(2500, 1000)

    cv2.imshow("MediaPipe Face Landmarker", frame)
    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:
        break
    if cv2.getWindowProperty("MediaPipe Face Landmarker", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyWindow("MediaPipe Face Landmarker")
vc.release()
