import cv2
import numpy as np
from PIL import Image
import os

def trainnmodel():
    # Initialize the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Path to the dataset
    path = "datasets"

    def getImageID(path):
        # List all image paths in the dataset folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        faces = []
        ids = []
        
        for imagePath in imagePaths:
            # Open image and convert to grayscale
            faceImage = Image.open(imagePath).convert('L')
            
            # Convert the grayscale image to a NumPy array
            faceNP = np.array(faceImage, 'uint8')
            
            # Extract ID from the filename
            try:
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
            except (IndexError, ValueError):
                print(f"Skipping file: {imagePath} (invalid ID format)")
                continue
            
            # Append face data and ID
            faces.append(faceNP)
            ids.append(Id)
            
            # Show the image during training (optional)
            cv2.imshow("Training", faceNP)
            cv2.waitKey(1)
        
        return ids, faces

    # Get the IDs and face data
    IDs, facedata = getImageID(path)

    # Train the recognizer
    if len(facedata) > 0:
        recognizer.train(facedata, np.array(IDs))
        recognizer.write("Trainer.yml")
        print("Training Completed............")
    else:
        print("No valid images found for training.")

    # Cleanup
    cv2.destroyAllWindows()


