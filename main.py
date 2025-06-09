from src.core.recognizer import HandGestureRecognizer

if __name__ == "__main__":
    app = HandGestureRecognizer("models/hand_gesture_model.pth")
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()