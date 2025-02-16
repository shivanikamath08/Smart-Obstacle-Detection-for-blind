import torch

# Load the model
model_path = "yolov5s.pt"  # Change this to your actual path
model = torch.load(model_path, map_location=torch.device('cpu'))

# Print model details
print(model.keys())  # Shows model components
