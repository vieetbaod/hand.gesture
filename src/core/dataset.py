from torch.utils.data import Dataset
from torch import FloatTensor
import numpy as np

class HandGestureDataset(Dataset):
    def __init__(self, landmarks, labels, sequence_length=30, transform=None):
        self.labels = labels
        self.landmarks = landmarks
        self.transform = transform
        self.sequence_length = sequence_length
        self.num_features = landmarks.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        landmark = self.landmarks[idx]
        label = self.labels[idx]
        
        # simple frame to sequence
        sequence = np.tile(landmark, (self.sequence_length, 1))
        
        # Convert to tensor
        sequence = FloatTensor(sequence)
        
        # transform
        if self.transform:
            sequence = self.transform(sequence)
        return sequence, label