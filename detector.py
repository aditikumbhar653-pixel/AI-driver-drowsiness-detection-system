import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import pygame

from db import get_connection


# ==========================================================
# PATHS AND SETTINGS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_model.yml"
)

ALARM_PATH = os.path.join(
    BASE_DIR,
    "static",
    "alarm.wav"
)

FACE_CONFIDENCE_LIMIT = 65

EAR_THRESHOLD = 0.21

EYE_CLOSED_SECONDS = 2.0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAMERA_INDEX = 0


# ==========================================================
# HAAR CASCADE
# ==========================================================

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_cascade.empty():

    print(
        "ERROR: Haar Cascade could not be loaded."
    )


# ==========================================================
# ALARM SETUP
# ==========================================================

alarm_loaded = False

try:

    pygame.mixer.init()

    if os.path.exists(ALARM_PATH):

        pygame.mixer.music.load(
            ALARM_PATH
        )

        pygame.mixer.music.set_volume(
            1.0
        )

        alarm_loaded = True

        print(
            "Alarm Loaded Successfully:",
            ALARM_PATH
        )

    else:

        print(
            "Alarm file not found:",
            ALARM_PATH
        )

except Exception as error:

    print(
        "Alarm setup error:",
        error
    )


# ==========================================================
# START ALARM
# ==========================================================

def start_alarm():

    if not alarm_loaded:
        return

    try:

        if not pygame.mixer.music.get_busy():

            pygame.mixer.music.play(
                loops=-1
            )

            print(
                "ALARM STARTED"
            )

    except Exception as error:

        print(
            "Alarm start error:",
            error
        )


# ==========================================================
# STOP ALARM
# ==========================================================

def stop_alarm():

    if not alarm_loaded:
        return

    try:

        if pygame.mixer.music.get_busy():

            pygame.mixer.music.stop()

    except Exception as error:

        print(
            "Alarm stop error:",
            error
        )


# ==========================================================
# SAVE REPORT TO MYSQL
# ==========================================================

def save_report(username, user_id, status):

    connection = None
    cursor = None

    try:

        connection = get_connection()

        if connection is None:
            print("Report error: MySQL connection failed.")
            return False

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO reports
            (
                driver_id,
                user_id,
                username,
                status,
                date_time,
                created_at
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                NOW(),
                NOW()
            )
            """,
            (
                int(user_id),
                int(user_id),
                username,
                status
            )
        )

        connection.commit()

        print(
            f"REPORT SAVED | "
            f"Driver ID: {user_id} | "
            f"User: {username} | "
            f"Status: {status}"
        )

        return True

    except Exception as error:

        print(
            "Report saving error:",
            error
        )

        if connection:

            try:
                connection.rollback()
            except Exception:
                pass

        return False

    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if connection:

            try:
                connection.close()
            except Exception:
                pass


# ==========================================================
# EYE LANDMARKS
# ==========================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ==========================================================
# DISTANCE
# ==========================================================

def distance(
    point1,
    point2
):

    return (
        (
            point1[0] - point2[0]
        ) ** 2
        +
        (
            point1[1] - point2[1]
        ) ** 2
    ) ** 0.5


# ==========================================================
# EYE ASPECT RATIO
# ==========================================================

def eye_ratio(
    landmarks,
    eye_points,
    width,
    height
):

    points = []

    for index in eye_points:

        landmark = landmarks[index]

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

    vertical1 = distance(
        points[1],
        points[5]
    )

    vertical2 = distance(
        points[2],
        points[4]
    )

    horizontal = distance(
        points[0],
        points[3]
    )

    if horizontal == 0:

        return 0

    return (
        vertical1 + vertical2
    ) / (
        2.0 * horizontal
    )
# ==========================================================
# LOAD FACE RECOGNIZER
# ==========================================================

def load_face_recognizer():

    # ------------------------------------------------------
    # Check OpenCV Face Module
    # ------------------------------------------------------

    if not hasattr(cv2, "face"):

        print(
            "ERROR: cv2.face is not available."
        )

        print(
            "Please install opencv-contrib-python."
        )

        return None

    # ------------------------------------------------------
    # Check Face Model
    # ------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print(
            "WARNING: Face model not found:"
        )

        print(
            MODEL_PATH
        )

        return None

    # ------------------------------------------------------
    # Load LBPH Model
    # ------------------------------------------------------

    try:

        recognizer = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        recognizer.read(
            MODEL_PATH
        )

        print(
            "Face model loaded successfully."
        )

        return recognizer

    except Exception as error:

        print(
            "Face model loading error:",
            error
        )

        return None


# ==========================================================
# PREPARE FACE
# ==========================================================

def prepare_face(
    gray_frame,
    face_rectangle
):

    x, y, w, h = face_rectangle

    # ------------------------------------------------------
    # Add margin around face
    # ------------------------------------------------------

    margin = int(
        0.15 * max(w, h)
    )

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        gray_frame.shape[1],
        x + w + margin
    )

    y2 = min(
        gray_frame.shape[0],
        y + h + margin
    )

    face = gray_frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:

        return None

    # ------------------------------------------------------
    # Resize for LBPH
    # ------------------------------------------------------

    face = cv2.resize(
        face,
        (200, 200)
    )

    return face


# ==========================================================
# MAIN CAMERA GENERATOR
# ==========================================================

def generate_frames(
    username="Unknown",
    user_id=0
):

    camera = None
    face_mesh = None

    print("")
    print(
        "========================================"
    )

    print(
        "STARTING CAMERA"
    )

    print(
        "Username:",
        username
    )

    print(
        "User ID:",
        user_id
    )

    print(
        "========================================"
    )

    # ======================================================
    # LOAD FACE RECOGNIZER
    # ======================================================

    recognizer = load_face_recognizer()

    # ======================================================
    # CHECK HAAR CASCADE
    # ======================================================

    if face_cascade.empty():

        print(
            "ERROR: Face cascade unavailable."
        )

        return

    try:

        # ==================================================
        # OPEN CAMERA
        # ==================================================

        print(
            "Opening camera index:",
            CAMERA_INDEX
        )

        camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        # --------------------------------------------------
        # Try normal backend if DirectShow fails
        # --------------------------------------------------

        if not camera.isOpened():

            print(
                "DirectShow camera open failed."
            )

            camera.release()

            camera = cv2.VideoCapture(
                CAMERA_INDEX
            )

        # --------------------------------------------------
        # Final camera check
        # --------------------------------------------------

        if not camera.isOpened():

            print(
                "ERROR: Camera could not be opened."
            )

            return

        # ==================================================
        # CAMERA SETTINGS
        # ==================================================

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        print(
            "CAMERA OPENED SUCCESSFULLY"
        )

        # ==================================================
        # MEDIAPIPE FACE MESH
        # ==================================================

        face_mesh = (
            mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        # ==================================================
        # VARIABLES
        # ==================================================

        eye_closed_start = None

        drowsiness_report_saved = False

        normal_report_saved = False

        frame_count = 0

        # ==================================================
        # CAMERA LOOP
        # ==================================================

        while True:

            success, frame = camera.read()

            # ------------------------------------------------
            # Camera frame check
            # ------------------------------------------------

            if not success:

                print(
                    "Unable to read camera frame."
                )

                break

            # ------------------------------------------------
            # Mirror camera
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            frame = cv2.resize(
                frame,
                (
                    FRAME_WIDTH,
                    FRAME_HEIGHT
                )
            )

            height, width = (
                frame.shape[:2]
            )

            # ------------------------------------------------
            # Convert to grayscale
            # ------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.equalizeHist(
                gray
            )

            # =================================================
            # FACE DETECTION
            # =================================================

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # =================================================
            # NO FACE DETECTED
            # =================================================

            if len(faces) == 0:

                stop_alarm()

                eye_closed_start = None

                drowsiness_report_saved = False

                normal_report_saved = False

                cv2.putText(
                    frame,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Monitoring Paused",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # =================================================
            # MULTIPLE FACES DETECTED
            # =================================================

            elif len(faces) > 1:

                stop_alarm()

                eye_closed_start = None

                drowsiness_report_saved = False

                normal_report_saved = False

                cv2.putText(
                    frame,
                    "Multiple Faces Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Only Registered Driver Allowed",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

                # Draw boxes around all faces
                for (
                    x,
                    y,
                    w,
                    h
                ) in faces:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

            # =================================================
            # EXACTLY ONE FACE
            # =================================================

            else:

                x, y, w, h = faces[0]

                # ------------------------------------------------
                # Prepare detected face
                # ------------------------------------------------

                face_crop = prepare_face(
                    gray,
                    faces[0]
                )

                matched_user = False

                confidence = 999

                predicted_label = -1

                # =================================================
                # FACE RECOGNITION
                # =================================================

                if (
                    face_crop is not None
                    and recognizer is not None
                ):

                    try:

                        (
                            predicted_label,
                            confidence
                        ) = recognizer.predict(
                            face_crop
                        )

                        frame_count += 1

                        # Print prediction every 30 frames
                        if frame_count % 30 == 0:

                            print(
                                "Predicted User ID:",
                                predicted_label
                            )

                            print(
                                "Confidence:",
                                round(
                                    confidence,
                                    2
                                )
                            )

                            print(
                                "Logged-in User ID:",
                                user_id
                            )

                        # ------------------------------------------------
                        # Match only logged-in user's face
                        # ------------------------------------------------

                        matched_user = (
                            int(
                                predicted_label
                            )
                            ==
                            int(user_id)
                            and
                            confidence
                            <=
                            FACE_CONFIDENCE_LIMIT
                        )

                    except Exception as error:

                        print(
                            "Face prediction error:",
                            error
                        )

                # =================================================
                # FACE MODEL NOT AVAILABLE
                # =================================================

                if recognizer is None:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 165, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face Model Not Ready",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Register Face First",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )

                    stop_alarm()

                    eye_closed_start = None

                    drowsiness_report_saved = False

                    normal_report_saved = False

                # =================================================
                # FACE NOT MATCHED
                # =================================================

                elif not matched_user:

                    stop_alarm()

                    eye_closed_start = None

                    drowsiness_report_saved = False

                    normal_report_saved = False

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face Not Matched",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Monitoring Paused",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # =================================================
                # REGISTERED USER MATCHED
                # =================================================

                else:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Registered Driver",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # =============================================
                    # MEDIAPIPE PROCESSING
                    # =============================================

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mesh_result = (
                        face_mesh.process(
                            rgb_frame
                        )
                    )

                    # =============================================
                    # CHECK EYE LANDMARKS
                    # =============================================

                    if (
                        mesh_result
                        .multi_face_landmarks
                    ):

                        landmarks = (
                            mesh_result
                            .multi_face_landmarks[0]
                            .landmark
                        )

                        # -----------------------------------------
                        # Calculate left eye EAR
                        # -----------------------------------------

                        left_ear = eye_ratio(
                            landmarks,
                            LEFT_EYE,
                            width,
                            height
                        )

                        # -----------------------------------------
                        # Calculate right eye EAR
                        # -----------------------------------------

                        right_ear = eye_ratio(
                            landmarks,
                            RIGHT_EYE,
                            width,
                            height
                        )

                        # -----------------------------------------
                        # Average EAR
                        # -----------------------------------------

                        ear = (
                            left_ear
                            +
                            right_ear
                        ) / 2.0

                        cv2.putText(
                            frame,
                            f"EAR: {ear:.2f}",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2
                        )

 # --------------------------------------------------
 # EYES CLOSED
 # --------------------------------------------------

                    if ear < EAR_THRESHOLD:

                        if eye_closed_start is None:
                            eye_closed_start = time.time()

                        closed_duration = (
                            time.time() - eye_closed_start
                        )

                        cv2.putText(
                            frame,
                            f"Eyes Closed: {closed_duration:.1f}s",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2
                        )

                        # ----------------------------------------------
                        # DROWSINESS DETECTED
                        # ----------------------------------------------

                        if closed_duration >= EYE_CLOSED_SECONDS:

                            start_alarm()

                            cv2.putText(
                                frame,
                                "DROWSINESS DETECTED!",
                                (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 0, 255),
                                3
                            )

                            cv2.putText(
                                frame,
                                "ALARM ACTIVE",
                                (20, 225),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                            # Save only one drowsiness report
                            if not drowsiness_report_saved:

                                save_report(
                                    username,
                                    user_id,
                                    "Drowsiness Detected"
                                )

                                drowsiness_report_saved = True
                                normal_report_saved = False

                    # --------------------------------------------------
                    # EYES OPEN
                    # --------------------------------------------------

                    else:

                        eye_closed_start = None

                        stop_alarm()

                        cv2.putText(
                            frame,
                            "Eyes Open - Normal",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        # Save normal report only once
                        if not normal_report_saved:

                            save_report(
                                username,
                                user_id,
                                "Normal"
                            )

                            normal_report_saved = True
                            drowsiness_report_saved = False

            # ------------------------------------------------------
            # CAMERA STATUS
            # ------------------------------------------------------

            cv2.putText(
                frame,
                "CAMERA ACTIVE",
                (
                    FRAME_WIDTH - 220,
                    FRAME_HEIGHT - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------------
            # CONVERT FRAME TO JPEG
            # ------------------------------------------------------

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not ret:
                print(
                    "ERROR: Frame encoding failed."
                )
                continue

            frame_bytes = buffer.tobytes()

            # ------------------------------------------------------
            # SEND FRAME TO FLASK
            # ------------------------------------------------------

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                +
                frame_bytes
                +
                b"\r\n"
            )

    except GeneratorExit:

        print(
            "Camera stream closed by browser."
        )

    except Exception as error:

        print(
            "Camera detection error:",
            error
        )

    finally:

        # ----------------------------------------------------------
        # STOP ALARM
        # ----------------------------------------------------------

        stop_alarm()

        # ----------------------------------------------------------
        # RELEASE MEDIAPIPE
        # ----------------------------------------------------------

        if face_mesh is not None:

            try:
                face_mesh.close()
            except Exception:
                pass

        # ----------------------------------------------------------
        # RELEASE CAMERA
        # ----------------------------------------------------------

        if camera is not None:

            try:
                camera.release()
            except Exception:
                pass

        print(
            "Camera released successfully."
        )

        print(
            "========================================"
        )
        print(
            "CAMERA STOPPED"
        )
        print(
            "========================================"
        )