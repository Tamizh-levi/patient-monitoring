import cv2
import os
from ultralytics import YOLO
from playsound import playsound
import threading
import time


# --- Configuration ---
class Config:
    BASE_DIR = os.getcwd()
    # Ensure you have the YOLO model file in your project directory
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
    # Path to a simple alert sound file
    ALERT_SOUND_PATH = os.path.join(BASE_DIR, "alarm_sounds", "fall_alert.mp3")

    # --- Detection Settings ---
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    # How long (in seconds) a patient must be out of bed before an alert is triggered
    ALERT_CONFIRMATION_TIME = 5



class BedExitDetector:
    def __init__(self, config):
        self.config = config
        self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
        self.bed_roi = None  # Will store the coordinates of the bed (x, y, w, h)

        # --- State Variables ---
        self.patient_out_of_bed_start_time = None
        self.alert_sent = False

    def trigger_alert(self):
        """Prints a message and plays a sound in a separate thread."""
        print(f"ALERT: Patient has been out of bed for over {self.config.ALERT_CONFIRMATION_TIME} seconds!")
        # Play sound in a non-blocking way
        if os.path.exists(self.config.ALERT_SOUND_PATH):
            threading.Thread(target=playsound, args=(self.config.ALERT_SOUND_PATH,), daemon=True).start()
        else:
            print(f"Warning: Alert sound not found at {self.config.ALERT_SOUND_PATH}")

    def select_bed_roi(self, frame):
        """Allows the user to select the bed's Region of Interest (ROI)."""
        print("\n" + "=" * 50)
        print("Please select the bed area with your mouse.")
        print("Draw a rectangle around the bed and press ENTER or SPACE.")
        print("Press 'c' to cancel the selection.")
        print("=" * 50 + "\n")

        # Use OpenCV's selectROI function
        roi = cv2.selectROI("Select Bed Area", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Bed Area")

        # Check if the user made a selection (width and height > 0)
        if roi[2] > 0 and roi[3] > 0:
            self.bed_roi = roi
            print(f"Bed area selected at: {self.bed_roi}")
        else:
            print("No bed area selected. Exiting.")
            exit()

    def run(self):
        """Main loop for the bed exit detection system."""

        # --- Using the IP camera URL you provided ---
        video_source = "http://10.225.186.214:3306/video"
        # To switch back to your computer's webcam, use: video_source = 0

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"Error: Cannot open camera source: {video_source}")
            print("Please check the URL and ensure your phone is on the same Wi-Fi network.")
            return

        # --- Step 1: Select the Bed ROI on the first frame ---
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Failed to capture the first frame.")
            cap.release()
            return
        self.select_bed_roi(first_frame)

        # --- Step 2: Main processing loop ---
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw the bed ROI on the frame for visualization
                x, y, w, h = self.bed_roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Bed Area", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Run YOLO detection
                results = self.yolo_model(frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)

                patient_in_bed = False
                for result in results:
                    for box in result.boxes:
                        # Check if the detected object is a person
                        if int(box.cls) == 0:
                            px1, py1, px2, py2 = map(int, box.xyxy[0])

                            # Draw the person's bounding box
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)

                            # Calculate the center of the person's bounding box
                            person_center_x = (px1 + px2) // 2
                            person_center_y = (py1 + py2) // 2

                            # Check if the person's center is inside the bed ROI
                            if x < person_center_x < (x + w) and y < person_center_y < (y + h):
                                patient_in_bed = True
                                cv2.putText(frame, "Patient in Bed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                            2)
                                break  # No need to check other people if one is in bed
                    if patient_in_bed:
                        break

                # --- Alert Logic ---
                if patient_in_bed:
                    # If patient is in bed, reset the timer and alert status
                    self.patient_out_of_bed_start_time = None
                    self.alert_sent = False
                else:
                    # If no patient is in bed, start the timer
                    cv2.putText(frame, "Patient NOT in Bed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if self.patient_out_of_bed_start_time is None:
                        self.patient_out_of_bed_start_time = time.time()

                    # If the timer has exceeded the threshold and no alert has been sent, trigger one
                    elapsed_time = time.time() - self.patient_out_of_bed_start_time
                    if not self.alert_sent and elapsed_time > self.config.ALERT_CONFIRMATION_TIME:
                        self.trigger_alert()
                        self.alert_sent = True  # Ensure alert only triggers once

                cv2.imshow("Bed Exit Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            print("Shutting down...")
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    config = Config()
    detector = BedExitDetector(config)
    detector.run()
