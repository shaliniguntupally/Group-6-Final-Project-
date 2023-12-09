from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pygame
import imutils
import time
import dlib
import cv2

def sound_alarm():
    # initialize the pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load("C:/Users/Shiva/alarm.wav")
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[14], mouth[18])
    B = dist.euclidean(mouth[12], mouth[16])
    mar = A / B
    return mar

def calculate_head_pose(shape):
    # Extract relevant landmarks for head pose estimation
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],   # Chin
        shape[45],  # Left eye left corner
        shape[36],  # Right eye right corner
        shape[54],  # Left Mouth corner
        shape[48]   # Right mouth corner
    ], dtype="double")

    # Camera parameters (change accordingly)
    # Camera parameters (change accordingly)
    focal_length = 950.0
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    # Distortion coefficients (change accordingly)
    dist_coeffs = np.zeros((4, 1))

    # Solve the PnP problem
    _, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    # Project a 3D point (nose end) onto the 2D image plane
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    return rotation_vector, translation_vector, nose_end_point2D

# Define constants
EYE_AR_THRESH = 0.3
MOUTH_AR_THRESH = 0.7
EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

# Object points for head pose estimation
object_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
], dtype="double")

# Metrics initialization
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Shiva/shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left, right eye, and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# Loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Head Pose Estimation
        rotation_vector, translation_vector, nose_end_point2D = calculate_head_pose(shape)

        for p in shape:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(shape[30][0]), int(shape[30][1]))
        p2 = (nose_end_point2D[0][0][0], nose_end_point2D[0][0][1])

        cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 255), 2)

        # Update metrics based on drowsiness detection
        if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
            if rect.width() < 100:  # Additional condition to check if the face is close enough for accurate detection
                continue

            # Drowsiness detected
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES or COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()

                # Update metrics
                true_positives += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Wakefulness detected
            COUNTER = 0
            ALARM_ON = False

            # Update metrics
            true_negatives += 1

        # Calculate metrics
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) != 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # Display metrics on the frame
        cv2.putText(frame, f"EAR: {ear:.2f} MAR: {mar:.2f} Head Pose: {rotation_vector[1][0]:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display drowsiness alert in a separate region
        if ALARM_ON:

          cv2.putText(frame, f"Accuracy: {accuracy:.2f} Precision: {precision:.2f} Recall: {recall:.2f} F1 Score: {f1_score:.2f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

          cv2.imshow("Frame", frame)
          key = cv2.waitKey(1) & 0xFF

          if key == ord("q"):
              break

cv2.destroyAllWindows()
vs.stop()


        
