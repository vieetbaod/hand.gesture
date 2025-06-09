from torch import cuda

EPOCHS = 50
BATCH_SIZE = 25
SEQUENCE_LENGTH = 30
SAVE_FOLDER = "models"
DEVICE = "cuda:0" if cuda.is_available() else "cpu"
NUM_FEATURES = 63  # 21 landmarks * 3 coordinates (x, y, z)
SAVE_PATH = f"{SAVE_FOLDER}/hand_gesture_model.pth"
OPTIM = {"alg": "adam", "lr": 0.001, "momentum": 0.5}
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
