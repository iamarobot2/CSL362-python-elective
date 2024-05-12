# Install necessary libraries
# pip install pillow opencv-python

import tkinter as tk
import PIL as pil
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Load the InceptionV3 model pre-trained on ImageNet data
image_model = InceptionV3(weights='imagenet')

# Load the YOLO model for object detection
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = yolo_net.getUnconnectedOutLayersNames()

class ImageRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Recognition App")

        # Create GUI elements
        self.label = tk.Label(self.master, text="Start the camera:")
        self.label.pack()

        self.camera_button = tk.Button(self.master, text="Start Camera", command=self.start_camera)
        self.camera_button.pack()

        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict_image)
        self.predict_button.pack()

        self.result_label = tk.Label(self.master, text="")
        self.result_label.pack()

        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        # Initialize camera
        self.cap = None
        self.camera_running = False

    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.cap = cv2.VideoCapture(0)  # Use camera with index 0 (you may need to change it based on your system)

            # Create a function to continuously update the camera feed
            def update():
                if self.camera_running:
                    ret, frame = self.cap.read()
                    if ret:
                        # Object detection
                        objects = self.detect_objects(frame)

                        # Image recognition
                        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_array = cv2.resize(img_array, (299, 299))
                        img_array = np.expand_dims(img_array, axis=0)

                        img_array = preprocess_input(img_array)
                        predictions = image_model.predict(img_array)
                        decoded_predictions = decode_predictions(predictions, top=3)[0]

                        # Display frame with detected objects
                        frame = self.draw_objects(frame, objects)
                        frame = self.draw_predictions(frame, decoded_predictions)

                        img = Image.fromarray(frame)
                        img = ImageTk.PhotoImage(img)

                        self.label.config(image=img)
                        self.label.image = img

                        self.master.after(10, update)

            update()
        else:
            self.stop_camera()

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(layer_names)

        height, width, _ = frame.shape
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([(x, y), (x + w, y + h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        objects = [(boxes[i][0], boxes[i][1]) for i in indices]

        return objects

    def draw_objects(self, frame, objects):
        for obj in objects:
            frame = cv2.rectangle(frame, obj[0], obj[1], (255, 0, 0), 2)
        return frame

    def draw_predictions(self, frame, decoded_predictions):
        frame = cv2.putText(frame, "Predictions:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        for i, (_, label, score) in enumerate(decoded_predictions):
            frame = cv2.putText(frame, f"{i + 1}: {label} ({score:.2f})", (10, 70 + 30 * i),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def predict_image(self):
        if self.camera_running:
            ret, frame = self.cap.read()
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, (299, 299))
            img_array = np.expand_dims(img_array, axis=0)

            img_array = preprocess_input(img_array)
            predictions = image_model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=3)[0]

            result_text = "Predictions:\n"
            for i, (_, label, score) in enumerate(decoded_predictions):
                result_text += f"{i + 1}: {label} ({score:.2f})\n"

            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Please start the camera first.")

    def quit_app(self):
        self.stop_camera()
        self.master.destroy()

# Create and run the GUI
root = tk.Tk()
app = ImageRecognitionApp(root)
root.mainloop()
