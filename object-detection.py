import torch
import cv2
import numpy as np
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Confidence threshold for detection
confidence_threshold = 0.6

# Camera parameters for distance calculation
KNOWN_WIDTH = 0.5  # Known object width in meters (e.g., an average person)
FOCAL_LENGTH = 615  # Example focal length in pixels (calibration needed)

# Text-to-speech function
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Calculate distance based on object width in the image
def calculate_distance(known_width, focal_length, width_in_image):
    if width_in_image == 0:  # Avoid division by zero
        return None
    return (known_width * focal_length) / width_in_image

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    speak("Camera not found. Please check the connection.")
    exit()

# Last speech time to prevent overlapping messages
last_speech_time = 0
speech_interval = 3  # Time in seconds

while True:
    ret, frame = cap.read()
    if not ret:
        speak("Camera feed interrupted.")
        break

    # Perform detection
    results = model(frame)

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
        distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)
        if distance:
            distance_text = f"{label}: {distance:.2f} meters"
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
        detected_objects.append(f"{label} at {distance:.2f} meters")

    # Provide voice guidance
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

    # Display the annotated frame
    cv2.imshow("Obstacle Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
