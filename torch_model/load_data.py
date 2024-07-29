import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    def __init__(self, np_file: str):
        scaling_factor = 6
        data = np.load(np_file, allow_pickle=True).item()
        images = np.array(data["images"])
        labels = data["labels"]

        embeddings = data["embeddings"]

        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings)

        embeddings = embeddings * scaling_factor
        self.images = images
        self.labels = labels
        self.embeddings = embeddings
        try:
            self.color_palette = data["color_palette"]
            self.color_palette_rgb = [
                tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
                for color in self.color_palette
            ]
        except KeyError:
            self.color_palette = data["hex_colors"]
            self.color_palette_rgb = [
                tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
                for color in self.color_palette
            ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        labels = self.labels[idx]
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        return img, labels, embeddings


def load_data(path, batch_size):
    dataset = NumpyDataset(path)
    color_palette = np.array(dataset.color_palette_rgb) / 255.0

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_set = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True
    )
    test_set = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True
    )

    return train_set, test_set, color_palette
