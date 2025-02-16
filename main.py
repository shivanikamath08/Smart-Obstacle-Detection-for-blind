import geopy
import pyttsx3
from navigation import get_directions, get_current_location
from obstacle_detection import detect_obstacles_and_guide
from speech_module import speak

def main():
    # Get user's current location or ask for start and end coordinates
    current_location = get_current_location()  # Get current location using GPS
    print(f"Current location: {current_location}")

    start_lat, start_lon = current_location  # Set current location as start (or ask user for coordinates)
    end_lat, end_lon = input("Enter destination coordinates (lat, lon): ").split(",")
    end_lat, end_lon = float(end_lat), float(end_lon)

    # Get directions and provide audio feedback
    directions = get_directions(start_lat, start_lon, end_lat, end_lon)
    speak(f"Start from your current location at {start_lat}, {start_lon}, heading towards {end_lat}, {end_lon}.")

    for direction in directions:
        speak(direction)

    # Start obstacle detection (Assuming webcam is used for real-time detection)
    detect_obstacles_and_guide()

if __name__ == "__main__":
    main()  