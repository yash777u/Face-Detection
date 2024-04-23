import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in a video stream
def detect_faces_webcam():
    # Open a connection to the webcam (usually 0 for the default webcam)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use cv2.CAP_DSHOW for DirectShow

    # Reduce frame size for faster processing
    frame_width = 640
    frame_height = 480
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    while True:
        # Capture video frame-by-frame
        ret, frame = cap.read()

        # Flip the frame horizontally (mirror)
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame with adjusted parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Add text on the frame - "Press 'Esc' to close the window"
        cv2.putText(frame, "Press 'Esc' to close the window", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Add text on the frame - "Web Cam Face Detection" at the upper-middle
        text_position = (int((frame_width - 200) / 2), 30)
        cv2.putText(frame, "Web Cam Face Detection", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Face Detection - Webcam', frame)

        # Introduce a delay to control frame rate (adjust as needed)
        if cv2.waitKey(10) == 27:  # 27 is the ASCII code for 'Esc'
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start face detection on the webcam
detect_faces_webcam()