import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import time
import mediapipe as mp

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1080x720")
        self.root.title("Attendance Management System")

        self.name_var = tk.StringVar()
        self.regd_var = tk.StringVar()
        self.password_var = tk.StringVar()

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.dataset_path = "dataset"
        self.attendance_file = "attendance.csv"
        self.registered_file = "registered_users.csv"

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            exit()

        self.create_widgets()
        self.check_files()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.add_user_tab = ttk.Frame(self.notebook)
        self.mark_attendance_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.add_user_tab, text="Add User")
        self.notebook.add(self.mark_attendance_tab, text="Mark Attendance")
        self.notebook.pack(expand=True, fill="both")

        ttk.Label(self.add_user_tab, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self.add_user_tab, textvariable=self.name_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.add_user_tab, text="Registration No:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(self.add_user_tab, textvariable=self.regd_var).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(self.add_user_tab, text="Password:").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(self.add_user_tab, textvariable=self.password_var, show="*").grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.add_user_tab, text="Add User", command=self.authenticate_and_add_user).grid(row=3, column=0, columnspan=2, pady=10)

        self.video_label = ttk.Label(self.mark_attendance_tab)
        self.video_label.grid(row=0, column=0)
        ttk.Button(self.mark_attendance_tab, text="Start Attendance", command=self.mark_attendance).grid(row=1, column=0)
        ttk.Button(self.mark_attendance_tab, text="Exit", command=self.exit_attendance).grid(row=1, column=1)

    def check_files(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        if not os.path.exists(self.registered_file):
            pd.DataFrame(columns=["Name", "RegdNo", "ID"]).to_csv(self.registered_file, index=False)
        if not os.path.exists(self.attendance_file):
            pd.DataFrame(columns=["ID", "Name", "RegdNo", "Timestamp"]).to_csv(self.attendance_file, index=False)

    def authenticate_and_add_user(self):
        password = self.password_var.get()
        if password != "soumya":
            messagebox.showerror("Error", "Authentication failed. Incorrect password.")
            return

        name = self.name_var.get()
        regd = self.regd_var.get()
        if not name or not regd:
            messagebox.showerror("Error", "Please fill all fields")
            return

        user_id = len(os.listdir(self.dataset_path)) + 1
        count = 0

        while count < 30:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image from webcam")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    count += 1
                    face_img = gray[y:y+height, x:x+width]
                    if face_img.size > 0:
                        cv2.imwrite(
                            f"{self.dataset_path}/User.{user_id}.{count}.jpg",
                            face_img
                        )

            cv2.imshow("Registering User", frame)
            if cv2.waitKey(1) == 27 or count >= 30:
                break

        cv2.destroyAllWindows()
        self.train_model()

        pd.DataFrame([[name, regd, user_id]], columns=["Name", "RegdNo", "ID"]).to_csv(
            self.registered_file, mode="a", header=False, index=False
        )

    def train_model(self):
        faces = []
        ids = []
        for image_path in os.listdir(self.dataset_path):
            if image_path.startswith("User."):
                img = cv2.imread(os.path.join(self.dataset_path, image_path), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    user_id = int(image_path.split(".")[1])
                    faces.append(img)
                    ids.append(user_id)
        if len(faces) > 0 and len(ids) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save("trainer.yml")

    def calculate_fps(self, prev_time):
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
        return fps, current_time

    def mark_attendance(self):
        if not os.path.exists("trainer.yml"):
            messagebox.showerror("Error", "Model not trained yet. Please add users first.")
            return

        self.display_frame = ttk.Frame(self.mark_attendance_tab)
        self.display_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.video_label = ttk.Label(self.display_frame)
        self.video_label.grid(row=0, column=0, padx=5)
        
        self.info_label = ttk.Label(self.display_frame, text="Waiting for detection...",
                                  font=("Arial", 12), justify="left")
        self.info_label.grid(row=0, column=1, padx=5, sticky="n")
        
        self.train_model()

        def update_frame():
            prev_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame")
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = False

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    face_roi = gray[y:y+height, x:x+width]
                    if face_roi.size > 0:
                        id_, confidence = self.recognizer.predict(face_roi)
                        if confidence < 70:
                            detected = True
                            user = pd.read_csv(self.registered_file)
                            user = user[user["ID"] == id_]
                            if not user.empty:
                                name = user["Name"].values[0]
                                regd_no = user["RegdNo"].values[0]
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                data = [[id_, name, regd_no, timestamp]]
                                pd.DataFrame(data).to_csv(
                                    self.attendance_file, mode="a", header=False, index=False
                                )
                                info_text = f"Name: {name}\nRegd No: {regd_no}\nTime: {timestamp}"
                                self.info_label.configure(text=info_text)
                                cv2.putText(frame, name, (x, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                          (0, 255, 0), 2)

                    cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)

            if not detected:
                self.info_label.configure(text="Waiting for detection...")

            fps, prev_time = self.calculate_fps(prev_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            img_resized = cv2.resize(frame, (640, 480))
            cv_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGBA)
            img_pil = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.video_label.after(10, update_frame)

        update_frame()

    def exit_attendance(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()