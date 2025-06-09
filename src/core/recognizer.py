from .constants import DEVICE, CLASSES, NUM_FEATURES, SEQUENCE_LENGTH
from torch import load as load_model, no_grad, FloatTensor
from .deep import HandGestureModel
from PIL import Image, ImageTk
import customtkinter as ctkt
import mediapipe as mp
import numpy as np
import cv2

class HandGestureInfo(ctkt.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.label = {"prd": None, "cfd": None}
        self.prd_t = self.cfd_t = None
        font = ctkt.CTkFont(
            family="Jetbrains Mono Medium",
            size=12,
        )

        for idx, var in enumerate(self.label.keys()):
            is_prd = var == "prd"
            temp = self.prd_t if is_prd else self.label[var]
            temp = ctkt.CTkLabel(self, font=font, compound="left", text=f"{"Class dự đoán" if is_prd else "Tỉ lệ dự đoán"}")
            temp.grid(row=0, column=idx if is_prd else idx, padx=10, pady=0, sticky="w")

            self.label[var] = ctkt.CTkTextbox(
                height=10,
                width=245,
                master=self,
                border_width=1,
                corner_radius=5,
                state="disabled",
                border_color="gray",
                fg_color="#2b2b2b",
            )
            self.label[var].grid(
                row=1,
                padx=10,
                pady=(0, 10),
                sticky="nsew",
                column=idx if is_prd else idx,
            )

    def update_text(self, text):
        for var in self.label.keys():
            self.label[var].configure(state="normal")
            self.label[var].delete("0.0", "end")
            self.label[var].insert("0.0", text[var])
            self.label[var].configure(state="disabled")

class HandGestureFrame(ctkt.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # create label for video display
        self.video = ctkt.CTkLabel(self, text="")
        self.video.grid(row=0, column=0, padx=10, pady=10)

    def update_frame(self, frame):
        # Convert frame to PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update label with new frame
        self.video.configure(image=photo)
        self.video.image = photo

class HandGestureRecognizer(ctkt.CTk):
    def __init__(self, model_path):
        super().__init__()
        # load model
        self.model = HandGestureModel(NUM_FEATURES).to(DEVICE)
        self.model.load_state_dict(load_model(model_path)["state_dict"])
        self.model.eval()

        # config mediapipe hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Configure window
        self.title("Hand Gesture Recognition")
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure(0, weight=1)

        # create frame
        self.frame = HandGestureFrame(master=self)
        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # create frame
        self.text = HandGestureInfo(master=self)
        self.text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # initialize camera
        self.capture = cv2.VideoCapture(0)

        # start video update
        self.start_video()

    def start_video(self):
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            frame = cv2.flip(frame, 1)

            frame = cv2.bilateralFilter(frame, 5, 50, 100)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = self.hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # get axis landmarks
                    arr_landmarks = []
                    for lm in hand_landmarks.landmark:
                        arr_landmarks.extend([lm.x, lm.y, lm.z])

                    # draw hand landmarks on the cropped frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(232, 254, 255), thickness=1, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(255, 249, 161), thickness=1, circle_radius=2)
                    )

                    # prediction
                    result = self.predict(arr_landmarks)
                    if result and ret:
                        self.text.update_text({"prd": result['class_name'], "cfd": result['confidence']})
            self.frame.update_frame(frame)
        # update after 10ms
        self.after(10, self.start_video)

    def on_closing(self):
        if self.capture.isOpened():
            self.capture.release()
        self.quit()

    def preprocess(self, landmarks):
        # create sequence
        sequence = np.tile(landmarks, (SEQUENCE_LENGTH, 1))
        # Convert to tensor
        tensor = FloatTensor(sequence)
        return tensor

    def predict(self, frame):
        tensor = self.preprocess(frame)

        # predict with cnn
        with no_grad():
            tensor = tensor.unsqueeze(0).to(DEVICE)
            outputs = self.model(tensor)
            predicted = outputs.argmax(dim=1)
            confidence = float(outputs.cpu().numpy()[0][predicted.item()] / outputs.sum().item())
            result = {
                'class_id': predicted.item(),
                'confidence': str(confidence)[:5],
                'class_name': CLASSES[predicted.item()]
            }
            print(f"Predict: Label: {result['class_name']} | Confidence: {result['confidence']}")
            return result