import cv2


def camera_test():
    """
    A simple script to test if the webcam can be opened by OpenCV.
    """
    print("Starting camera test...")
    found_camera = False
    for i in range(5):
        print(f"Attempting to open camera with index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found_camera = True
            print(f"Success! Camera opened with index {i}. Press 'q' to exit.")

            # Read and display frames from the working camera
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame.")
                    break

                # Display the frame in a window
                cv2.imshow(f"Camera Test (Index {i})", frame)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            break  # Exit the loop after finding and testing one camera

    if not found_camera:
        print("Error: No camera could be opened. Please check the following:")
        print("1. Is your camera plugged in and working?")
        print("2. Are any other applications (like Zoom, Teams) using the camera?")
        print("3. Check your Windows 'Camera privacy settings' to ensure Python has access.")


if __name__ == "__main__":
    camera_test()