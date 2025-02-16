import cv2
import torch
import pyttsx3
import time
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut

# Initialize pyttsx3 for speech synthesis
engine = pyttsx3.init()

# Confidence threshold for detection
confidence_threshold = 0.6

# Camera parameters for distance calculation
KNOWN_WIDTH = 0.5  # Known object width in meters (e.g., an average person)
FOCAL_LENGTH = 615  # Example focal length in pixels (calibration needed)

# Initialize geolocator
geolocator = Nominatim(user_agent="ObstacleDetectionApp")

# Text-to-speech function
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Function to get latitude and longitude from a location name
def get_location_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            speak(f"Sorry, I couldn't find the location: {location_name}")
            return None, None
    except GeocoderTimedOut:
        speak("Geocoding service timed out. Please try again.")
        return None, None

# Calculate distance based on object width in the image
def calculate_distance(known_width, focal_length, width_in_image):
    if width_in_image == 0:  # Avoid division by zero
        return None
    return (known_width * focal_length) / width_in_image

def detect_obstacles_and_guide():
    # Load YOLOv5 model using torch.hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load the pretrained YOLOv5s model

    # Capture video from webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not cap.isOpened():
        speak("Camera not found. Please check the connection.")
        exit()

    # Initialize timers
    last_speech_time = 0
    last_distance_message_time = 0
    speech_interval = 3  # Time interval for obstacle messages (seconds)
    distance_message_interval = 30  # Time interval for distance messages (seconds)

    # Get current and destination locations
    current_location = input("Enter your current location: ")
    current_lat, current_lon = get_location_coordinates(current_location)
    if current_lat is None or current_lon is None:
        return

    destination_location = input("Enter your destination location: ")
    destination_lat, destination_lon = get_location_coordinates(destination_location)
    if destination_lat is None or destination_lon is None:
        return

    # Calculate initial distance to the destination
    start = (current_lat, current_lon)
    end = (destination_lat, destination_lon)
    distance = geodesic(start, end).km

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Camera feed interrupted.")
            break

        # Use YOLOv5 to predict objects in the frame
        results = model(frame)  # Get predictions from the YOLOv5 model
        
        # Annotate frame with results
        detections = results.pandas().xyxy[0]  # Get detection results as a pandas DataFrame
        annotated_frame = np.squeeze(results.render())  # Render annotated frame

        # Guidance flags
        obstacle_left = False
        obstacle_center = False
        obstacle_right = False
        detected_objects = []

        for _, row in detections.iterrows():
            confidence = row['confidence']
            if confidence < confidence_threshold:
                continue

            label = row['name']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Calculate bounding box width and distance
            box_width = x2 - x1
            object_distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)
            if object_distance:
                distance_text = f"{label}: {object_distance:.2f} meters"
                cv2.putText(annotated_frame, distance_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Determine position in the frame
            frame_width = frame.shape[1]
            if x1 < frame_width / 3:
                obstacle_left = True
            elif x2 > 2 * frame_width / 3:
                obstacle_right = True
            else:
                obstacle_center = True

            # Add detected objects for audio feedback
            detected_objects.append(f"{label} at {object_distance:.2f} meters")

        # Provide voice guidance for obstacles
        current_time = time.time()
        if detected_objects and current_time - last_speech_time > speech_interval:
            obstacle_message = ", ".join(detected_objects)
            if obstacle_center:
                speak(f"Obstacle ahead. {obstacle_message}")
            elif obstacle_left:
                speak(f"Obstacle on the left. {obstacle_message}")
            elif obstacle_right:
                speak(f"Obstacle on the right. {obstacle_message}")
            last_speech_time = current_time

        # Provide distance message every 30 seconds
        if current_time - last_distance_message_time > distance_message_interval:
            speak(f"Your destination is {distance:.2f} kilometers away.")
            last_distance_message_time = current_time

        # Display the annotated frame
        cv2.imshow("Obstacle Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_obstacles_and_guide()
