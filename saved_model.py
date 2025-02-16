import torch

# Load trained model
model = torch.load("yolov5su.pt", map_location=torch.device('cpu'))

# Print success message
print("Model loaded successfully!")
