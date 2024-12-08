# Face Recognition System

## Overview
This is a Python-based face recognition system using OpenCV that allows for face data collection, model training, and face recognition.

## Prerequisites
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pillow (PIL)

## Project Structure
- `dataacollect.py`: Data collection script
- `trainingdemo.py`: Model training script
- `testmodel.py`: Face recognition testing script

## Step 1: Data Collection (`dataacollect.py`)

### Purpose
The `dataacollect.py` script is responsible for collecting face images for training the face recognition model.

### How to Use
1. Run the script
2. Enter your personal details:
   - User ID (numeric identifier)
   - Name

### Features
- Uses webcam to capture face images
- Automatically detects and crops faces
- Saves images in the `datasets` folder
- Validates user ID and name to prevent duplicates
- Collects up to 1000 face images for training

### Important Notes
- Ensure your webcam is connected before running
- Face must be clearly visible to the camera
- Press 'q' to exit manually
- The script automatically starts model training after collecting images

### Requirements
- Haar Cascade XML file for face detection
- Webcam access

### Validation Checks
- Checks for existing user IDs
- Validates image capture
- Ensures proper face detection

## Troubleshooting
- Verify webcam connectivity
- Check Haar Cascade file path
- Ensure good lighting and face visibility

## Next Steps
After running this script, proceed to model training using `trainingdemo.py`
