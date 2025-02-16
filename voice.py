import pyttsx3
import requests
import speech_recognition as sr
import geocoder
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import cv2
import torch
import numpy as np
import time

# Initialize text-to-speech
engine = pyttsx3.init()

# Confidence threshold for detection
confidence_threshold = 0.2

# Camera parameters for distance calculation
KNOWN_WIDTH = 0.5  # Known object width in meters (e.g., an average person)
FOCAL_LENGTH = 615  # Example focal length in pixels (calibration needed)

# GraphHopper API Key (Replace with your key)
GRAPHHOPPER_API_KEY = "71dd7179-cfbb-4115-a314-398695b35742"
GRAPHHOPPER_API_URL = "https://graphhopper.com/api/1/route"

# Initialize geolocator
geolocator = Nominatim(user_agent="ObstacleDetectionApp")

# Function to speak text
def speak(message):
    print(message)  # Debugging purpose
    engine.say(message)
    engine.runAndWait()

# Get latitude and longitude from location name
def get_location_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name, exactly_one=True)
        if location:
            return location.latitude, location.longitude
        else:
            speak(f"Could not find {location_name}. Please provide a more specific location.")
            return None, None
    except GeocoderTimedOut:
        speak("Geocoding service timed out. Please try again.")
        return None, None

# Get total distance using GraphHopper API
def get_distance(start_coords, end_coords):
    if not start_coords or not end_coords:
        speak("Invalid coordinates. Please check your locations.")
        return None
    
    try:
        params = {
            'point': [f"{start_coords[0]},{start_coords[1]}", f"{end_coords[0]},{end_coords[1]}"],
            'type': 'json',
            'locale': 'en',
            'vehicle': 'foot',  
            'key': GRAPHHOPPER_API_KEY
        }
        response = requests.get(GRAPHHOPPER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if "paths" in data and len(data["paths"]) > 0:
            return data["paths"][0]["distance"] / 1000  # Convert meters to km
        else:
            speak("No route found. Please check your locations.")
            return None
    except requests.exceptions.RequestException as e:
        speak(f"Error fetching distance: {e}")
        return None

# Function to take voice input for destination
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Please say your destination.")
        print("Listening for destination...")
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Reduce background noise
        try:
            audio = recognizer.listen(source, timeout=10)  
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Try again.")
            return None
        except sr.RequestError:
            speak("Speech recognition service is unavailable.")
            return None
        except sr.WaitTimeoutError:
            speak("No response detected.")
            return None

# Function to get current location coordinates
def get_current_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng[0], g.latlng[1]
        else:
            speak("Could not fetch current location. Using default coordinates.")
            return (12.3051, 76.6551)  
    except Exception as e:
        speak(f"Error fetching current location: {e}")
        return None, None

# Calculate distance based on object width in the image
def calculate_distance(known_width, focal_length, width_in_image):
    if width_in_image <= 0:  
        return None  # Avoid division by zero
    return (known_width * focal_length) / width_in_image

# Detect obstacles and provide real-time feedback
def detect_obstacles_and_guide(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera access failed. Please check your camera.")
        return

    last_speak_time = 0  
    speak_interval = 2  # Speak at most every 2 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Camera error, please check the connection.")
            break

        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        results = model(frame_resized)

        # Process detections
        for det in results.xyxy[0]:  
            x_min, y_min, x_max, y_max, conf, cls = det.tolist()
            if conf >= confidence_threshold:  
                width_in_image = x_max - x_min
                distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, width_in_image)

                center_x = (x_min + x_max) // 2  
                frame_center = frame.shape[1] // 2  
                direction = "Move Left" if center_x > frame_center else "Move Right"

                class_id = int(cls)
                detected_object = model.names[class_id]  

                if distance and (time.time() - last_speak_time > speak_interval):
                    speak(f"{detected_object} detected at {distance:.2f} meters. {direction}.")
                    last_speak_time = time.time()  # Update last speak time

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_object} ({distance:.2f}m)", (int(x_min), int(y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Obstacle Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Slight delay to avoid lag
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  

    current_coords = get_current_location()
    if not current_coords:
        speak("Error fetching current location. Exiting.")
        exit()
    
    destination = None
    while not destination:
        destination = get_voice_input()
        if not destination:
            speak("Please try again.")
    
    destination_coords = get_location_coordinates(destination)
    if not destination_coords:
        speak("Could not determine destination coordinates. Exiting.")
        exit()
    
    total_distance = get_distance(current_coords, destination_coords)
    if total_distance is not None:
        speak(f"The total distance from your current location to {destination} is {total_distance:.2f} kilometers.")
    else:
        speak("Could not fetch the distance. Please try again later.")

    detect_obstacles_and_guide(model)
