import cv2
import torch
import pyttsx3
import time
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Initialize pyttsx3 for speech synthesis
engine = pyttsx3.init()

# Confidence threshold for detection
confidence_threshold = 0.2

# Camera parameters for distance calculation
KNOWN_WIDTH = 0.5  # Known object width in meters (e.g., an average person)
FOCAL_LENGTH = 615  # Example focal length in pixels (calibration needed)

# Initialize geolocator
geolocator = Nominatim(user_agent="ObstacleDetectionApp")

# GraphHopper API URL for routing
GRAPHHOPPER_API_URL = "https://graphhopper.com/api/1/route"

# Replace with your actual GraphHopper API Key
GRAPHHOPPER_API_KEY = "b362adab-c597-4718-8b9c-aac65cc5ec59"

# Text-to-speech function
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Get latitude and longitude from a location name
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

# Get directions and total distance between two points using GraphHopper API
def get_directions(start, end):
    start_lat, start_lon = get_location_coordinates(start)
    end_lat, end_lon = get_location_coordinates(end)

    if not start_lat or not start_lon or not end_lat or not end_lon:
        return None, None

    params = {
        'point': [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
        'type': 'json',
        'locale': 'en',
        'vehicle': 'foot',  # 'foot' is used for walking; can be changed to 'car', 'bike', etc.
        'key': GRAPHHOPPER_API_KEY
    }

    response = requests.get(GRAPHHOPPER_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()

        if "paths" in data and len(data["paths"]) > 0:
            total_distance = data["paths"][0]["distance"] / 1000  # Convert meters to kilometers
            return total_distance
        else:
            speak("Unable to fetch directions. Please try again later.")
            return None
    else:
        speak("Error connecting to directions service.")
        return None

# Calculate distance based on object width in the image
def calculate_distance(known_width, focal_length, width_in_image):
    if width_in_image == 0:  # Avoid division by zero
        return None
    return (known_width * focal_length) / width_in_image

# Detect obstacles and provide directional guidance
def detect_obstacles_and_guide():
    # Load YOLOv5 model using torch.hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load the pretrained YOLOv5s model

    # Capture video from webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not cap.isOpened():
        speak("Camera not found. Please check the connection.")
        exit()

    # Starting location is Canara Engineering College
    start_location = "Vidyavardhaka college of engineering"
    speak(f"The starting location is set to {start_location}.")

    # Get destination from user
    destination_location = input("Enter your destination location: ").strip()
    speak(f"You entered your destination as {destination_location}.")
    destination_lat, destination_lon = get_location_coordinates(destination_location)
    if destination_lat is None or destination_lon is None:
        return

    # Get total distance
    total_distance = get_directions(start_location, destination_location)
    if total_distance is None:
        return
    speak(f"The total distance to your destination is {total_distance:.2f} kilometers. Starting obstacle detection and guidance.")

    # Start obstacle detection
    guidance_cooldown = time.time()  # Initialize cooldown timer
    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Camera feed interrupted.")
            break

        # Use YOLOv5 to predict objects in the frame
        results = model(frame)  # Get predictions from the YOLOv5 model
        detections = results.pandas().xyxy[0]  # Get detection results as a pandas DataFrame
        annotated_frame = np.squeeze(results.render())  # Render annotated frame

        frame_center_x = frame.shape[1] // 2  # Get the x-coordinate of the frame's center
        detected_objects = []  # Store detected objects and guidance

        for _, row in detections.iterrows():
            confidence = row['confidence']
            if confidence < confidence_threshold:
                continue

            label = row['name']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Calculate bounding box width and object position
            box_width = x2 - x1
            object_center_x = (x1 + x2) // 2  # Center of the object in the frame
            object_distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)

            if object_distance:
                # Determine direction based on object's position
                if object_center_x < frame_center_x:
                    direction = "Move right to avoid"
                else:
                    direction = "Move left to avoid"

                distance_text = f"{label}: {object_distance:.2f} meters"
                guidance_text = f"{direction} {label} at {object_distance:.2f} meters"

                # Display text on the frame
                cv2.putText(annotated_frame, distance_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                detected_objects.append((guidance_text, object_distance))

        # Provide voice guidance for obstacles within 2 meters and a cooldown period
        current_time = time.time()
        for guidance_text, object_distance in detected_objects:
            if object_distance <= 2 and (current_time - guidance_cooldown > 3):  # 3-second cooldown
                speak(guidance_text)
                guidance_cooldown = current_time
                break  # Speak only once for the closest relevant object

        # Display the annotated frame
        cv2.imshow("Obstacle Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    speak("Obstacle detection stopped.")
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_obstacles_and_guide()
