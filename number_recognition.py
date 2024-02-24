import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def calculate_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def main():

    # Create an GestureRecognizer object
    base_options = python.BaseOptions(model_asset_path="numbers.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        ret, frame = cap.read()

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process the frame
        results = recognizer.recognize(mp_image)
        # print(results)

        # for each hand in the frame 
        if results.hand_landmarks is not None:
            for hand_landmarks in results.hand_landmarks:
                # Draw the landmarks
                print(hand_landmarks)
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw the bounding box
                bounding_rect = calculate_bounding_rect(frame, hand_landmarks)
                cv2.rectangle(frame, (bounding_rect[0], bounding_rect[1]), (bounding_rect[2], bounding_rect[3]), (0, 255, 0), 2)

                # Draw the gesture
                score = results.gestures[0][0].score
                category_name = results.gestures[0][0].category_name

                cv2.putText(frame, f"{category_name} ({score:.2f})", (bounding_rect[0], bounding_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Show the frame
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()