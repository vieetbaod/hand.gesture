from src.core.constants import CLASSES
import mediapipe as mp
from csv import writer
from tqdm import tqdm
import pandas as pd
import cv2
import os

def draw_images_landmark(dataset_path, save_path):
    """
        Thu thập landmark bàn tay từ ảnh và lưu các tọa độ x, y, z của các điểm landmarks vào file .csv
        Args:
             dataset_path (str): Đường dẫn tới dataset ảnh, tổ chức theo thư mục nhãn
             save_path (str): Đường dẫn lưu file .csv
        Returns:
             None
    """
    # mediapipe hands instance
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    # define hand solution
    with mp_hands.Hands(
            static_image_mode=True,  # Chế độ nhận diện ảnh tĩnh
            max_num_hands=1,  # Tối đa 1 bàn tay mỗi ảnh
            min_detection_confidence=0.5  # Ngưỡng tự tin phát hiện
    ) as hands:
        # input, output paths
        input_path = os.path.join("data", dataset_path)
        output_path = os.path.join("data", save_path)
        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, "landmarks.csv")
        # Prepare CSV header
        header = ["label"] + [f"x{i},y{i},z{i}" for i in range(21)]
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = writer(csvfile)
            csv_writer.writerow(header)
            for label_idx in tqdm(range(len(CLASSES)), desc="Labels"):
                label = CLASSES[label_idx]
                label_path = os.path.join(input_path, label)
                image_names = [img for img in os.listdir(label_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for image_name in tqdm(image_names, desc=f"Processing {label} | {label_idx}", leave=False):
                    image_path = os.path.join(label_path, image_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    # convert image to BGR if it is grayscale or has alpha channel
                    if len(image.shape) == 2 or image.shape[2] == 1:  # grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.shape[2] == 4:  # alpha channel
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
                    result = hands.process(img_rgb)  # hand detection

                    # if hands are detected
                    if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])
                            row = [label_idx] + landmarks
                            csv_writer.writerow(row)
                            annotated_image = image.copy()
                            mp_draw.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            cv2.imshow('Hand Detection', annotated_image)
                            cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("Completed save to ", csv_path)


if __name__ == "__main__":
    # draw_images_landmark("images", "csv")
    csv = pd.read_csv("data/csv/landmarks.csv")
    print(csv.head())
    print(csv.shape)
