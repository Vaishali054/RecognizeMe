# Real-time Emotion Recognition

This project demonstrates real-time emotion recognition using a webcam and an 5 layer architecture model trained on the FER-2013 dataset.

## Overview

The project uses computer vision techniques and deep learning to recognize facial expressions in real-time. It leverages OpenCV for webcam input and processing, and PyTorch for the LSTM model implementation.

## Requirements

- Python 3.x
- PyTorch
- OpenCV (cv2)
- torchvision
- NumPy

Install the required packages using the following command:

   ```bash
   pip install torch torchvision opencv-python numpy
   ```

## Project Structure

The project structure is organized as follows:

```
realtime_emotion.py         # Main script for real-time emotion recognition using webcam
models/
└── lstm.py                 # LSTM model architecture
best_model.pth              # Trained model weights (not included in this repository)
README.md                   # Project documentation (you are here)
```

## Usage

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd RECOGNIZEME
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained model weights**:

   You need to download the `best_model.pth` file containing the trained LSTM model weights. You can obtain these weights by training the model using the `main.py` script or by acquiring them from your project supervisor.

4. **Run the real-time emotion recognition script**:

   ```bash
   python realtime_emotion.py
   ```

   Press `q` to exit the application.

## Model

The LSTM model used in this project is implemented in PyTorch and is trained to recognize emotions from facial images. The model is trained on the FER-2013 dataset and achieves good accuracy for basic emotions such as Happy, Sad, Angry, etc. The model uses a 5 layer architecture which was descried in research paper named "Facial Emotion Recognition using CNN with Data Augmentation".

## Implementation Details

### Training the Model

To train the model:

1. Ensure you have the FER-2013 dataset or access to it.
2. Use the `main.py` script to train the model. Update the script with your dataset paths and run:

   ```bash
   python main.py
   ```

## Notes

- Ensure your webcam is properly connected and accessible.
- This project assumes basic knowledge of Python, PyTorch, and OpenCV.
- The project is intended for educational purposes and may require further optimization for production-level deployment.