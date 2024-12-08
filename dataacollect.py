import cv2
import os
from trainingdemo import trainnmodel

def dataacollect():   
    # Path to Haar Cascade
    cascade_path = r"Z:\TO DO\codes\intrusive_thoughts\face_recognition\face_recognition_and_door_lock\haarcascade_frontalface_default.xml"

    # Load Haar Cascade
    facedetect = cv2.CascadeClassifier(cascade_path)

    # Check if Haar Cascade is loaded properly
    if facedetect.empty():
        print("Error: Haar Cascade file not found or corrupted. Please check the file path.")
        exit()

    # Video source
    video = cv2.VideoCapture(0)

    # Check if video file was successfully opened
    if not video.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Ensure the dataset folder exists
    os.makedirs("datasets", exist_ok=True)

    # Get user ID and name for dataset
    id = int(input("Enter Your ID: "))
    name = input("Enter your Name: ")

    # Function to validate ID and name
    def validate_id_name(name, id):
        id = str(id) 
        existing_files = os.listdir("datasets")
        
        for file in existing_files:
            parts = file.rsplit(".", maxsplit=3)
            
            # Ensure valid file format: <name>.<id>.<count>.jpg
            if len(parts) == 4:  
                existing_name = parts[0]
                existing_id = parts[1]
                existing_name = existing_name.strip()  # Remove any leading or trailing spaces
                name = name.strip()  
                if int(existing_id) == int(id):
                    if existing_name == name:
                        print("You are already registered. If this is incorrect, please change your ID.")
                    else:
                        print(f"ID {id} is already assigned to {existing_name}. Please choose a different ID.")
                    dataacollect()
                    exit()

    # Validate ID and name
    try:
        validate_id_name(name, id)
    except ValueError as e:
        print(e)
        exit()

    count = 0

    while True:
        ret, frame = video.read()

        # Validate the captured frame
        if not ret or frame is None:
            print("Error: Unable to read frame. Exiting...")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        # Process detected faces
        for (x, y, w, h) in faces:
            count += 1
            # Save the cropped face as an image
            cv2.imwrite(f'datasets/{name}.{id}.{count}.jpg', gray[y:y+h, x:x+w])
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        # Display the frame with detected faces
        cv2.imshow("Frame", frame)

        # Break the loop if the required number of images is collected
        if count >= 1000:
            print("Dataset collection completed!")
            trainnmodel()
            break

        # Press 'q' to quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()


dataacollect()