import os
import sys
import cv2  #video rendering and image processing
import dlib  #face and landmark detection
import imutils  #image resizing and processing
from scipy.spatial import distance as dist  #calculating distances between landmarks
from imutils import face_utils  # For facial landmark utilities

VIDEO_PATH = os.path.join('assets', 'my_blink.mp4')
MODEL_PATH = os.path.join('Model', 'shape_predictor_68_face_landmarks.dat')
BLINK_THRESHOLD = 0.2  #detect a blink
CONSECUTIVE_FRAMES = 2  #no. of consecutive frames the EAR must be below threshold

def calculate_EAR(eye):
    #euclidean distances between the vertical eye landmarks
    vertical1 = dist.euclidean(eye[1], eye[5])
    vertical2 = dist.euclidean(eye[2], eye[4])

    #euclidean distance between the horizontal eye landmarks
    horizontal = dist.euclidean(eye[0], eye[3])

    #calculation of EAR
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

def initialize_video_capture(video_path):
    #initialize video capture from the given path
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)
    return cam

def initialize_landmark_detector(model_path):
    #initialize dlib's face detector and landmark predictor
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(model_path)
    return detector, landmark_predictor

def main():
    #initialize video capture
    cam = initialize_video_capture(VIDEO_PATH)

    #initialize face detector and landmark predictor
    detector, landmark_predictor = initialize_landmark_detector(MODEL_PATH)

    #get the indexes of the facial landmarks for the left and right eye
    (L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    #initialize counters
    blink_count = 0
    frame_counter = 0

    print("Starting Blink Detection. Press 'q' to exit.")

    while True:
        #read the next frame from the video
        ret, frame = cam.read()

        #break the loop if the video has ended
        if not ret:
            print("Video ended. Exiting...")
            break

        #resize the frame for faster processing
        frame = imutils.resize(frame, width=640)

        #convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect faces in the grayscale frame
        faces = detector(gray, 0)

        #loop over the detected faces
        for face in faces:
            #determine the facial landmarks for the face region
            shape = landmark_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            #extract the left and right eye coordinates
            left_eye = shape[L_START:L_END]
            right_eye = shape[R_START:R_END]

            #compute the EAR for both eyes
            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)

            #average the EAR together
            avg_EAR = (left_EAR + right_EAR) / 2.0

            #check if the EAR is below the blink threshold
            if avg_EAR < BLINK_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                    cv2.putText(frame, 'Blink Detected', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                frame_counter = 0

            #draw the eye regions for visualization
            leftEyeHull = cv2.convexHull(left_eye)
            rightEyeHull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            #display the EAR value on the frame
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #display the frame with annotations
        cv2.imshow("Blink Detection", frame)

        #exit the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    #release the video capture and close all windows
    cam.release()
    cv2.destroyAllWindows()
    print(f"Total Blinks Detected: {blink_count}")


if __name__ == "__main__":
    main()
