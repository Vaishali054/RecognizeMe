import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from models.lstm import LSTMModel
import numpy as np

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model parameters
input_size = 48 * 48  # each image is 48x48 pixels
hidden_size = 128
num_layers = 2
num_classes = 7  # Number of emotions

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)

# Load the trained model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define transformations for the input data
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to predict emotion from a face image
def predict_emotion(face_image):
    # Apply transformations
    img = data_transform(face_image).unsqueeze(0).to(device)
    
    # Predict emotion
    with torch.no_grad():
        model.eval()
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()

# OpenCV setup for webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get face coordinates using Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces and predict emotion for each face
    for (x, y, w, h) in faces:
        # Draw rectangle around faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face
        face = gray[y:y+h, x:x+w]
        
        # Predict emotion
        predicted_class = predict_emotion(face)
        
        # Assign label to the emotion
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion = emotions[predicted_class]
        
        # Write emotion text above rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Recognition', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
