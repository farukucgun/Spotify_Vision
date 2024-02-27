import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import requests

last_action_time = 0
spotify_access_token = ""


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


def take_action(spotify_access_token, category_name):
    print(f"Detected number: {category_name}")

    if category_name == "1":
        play_song(spotify_access_token, "Somewhere I Belong")

    elif category_name == "2":
        get_playback_info(spotify_access_token)


def login_to_spotify():
    # Get the access token using authorization_code grant type
    print("Logging in to Spotify...")
    response = requests.get("https://accounts.spotify.com/authorize", params={ 
        "client_id": config.spotify_client_id,
        "response_type": "code",
        "redirect_uri": config.redirect_uri,
        "scope": "user-read-playback-state user-modify-playback-state",
    })

    # go to the url, authenticate and get the callback url


def get_auth_header(spotify_access_token):
    return {"Authorization": f"Bearer {spotify_access_token}"}


def get_playback_info(spotify_access_token):
    # Get the playback info
    response = requests.get("https://api.spotify.com/v1/me/player", headers=get_auth_header(spotify_access_token))
    print(response.json())
    return response.json()


def play_song(spotify_access_token, song_name):
    # Search for the song
    response = requests.get(f"https://api.spotify.com/v1/search?q={song_name}&type=track", headers=get_auth_header(spotify_access_token))

    song_uri = response.json()["tracks"]["items"][0]["uri"]

    # Play the song
    res = requests.put("https://api.spotify.com/v1/me/player/play", json={
        "uris": [song_uri]
    }, headers=get_auth_header(spotify_access_token))

    print(res)


def main():
    global last_action_time
    global spotify_access_token

    # Create an GestureRecognizer object
    base_options = python.BaseOptions(model_asset_path="numbers.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    spotify_access_token = login_to_spotify()

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

        # Check if the hand landmarks are detected
        if results.hand_landmarks is not None:
            for hand_landmarks in results.hand_landmarks:
                # Draw the landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw the bounding box
                bounding_rect = calculate_bounding_rect(frame, hand_landmarks)
                cv2.rectangle(frame, (bounding_rect[0], bounding_rect[1]), (bounding_rect[2], bounding_rect[3]), (0, 255, 0), 2)

                # Draw the gesture category and score
                score = results.gestures[0][0].score
                category_name = results.gestures[0][0].category_name

                cv2.putText(frame, f"{category_name} ({score:.2f})", (bounding_rect[0], bounding_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                # Check if the gesture is a number
                if score > 0.5 and category_name in config.valid_gestures:
                    # Check if the action timeout has passed
                    if (cv2.getTickCount() - last_action_time) / cv2.getTickFrequency() > config.action_timeout:
                        # Take the action
                        take_action(spotify_access_token, category_name)

                        # Update the last action time
                        last_action_time = cv2.getTickCount()

        # Show the frame
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()