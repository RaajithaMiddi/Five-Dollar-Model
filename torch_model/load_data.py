import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset


class ImageDataset(TensorDataset):
    def __init__(self, np_file: str, scaling_factor=6):
        data = np.load(np_file, allow_pickle=True).item()

        self.images = np.asarray(data["images"], dtype=np.float32)
        self.embeddings = np.asarray(data["embeddings"], dtype=np.float32) * scaling_factor

        super().__init__(
            torch.from_numpy(self.images),
            torch.from_numpy(self.embeddings)
        )

        self.ids = data.get('ids')
        self.categories = data.get("categories")
        self.labels = data["labels"]

        palette_rgb = data.get("rgb_colors")
        palette_hex = data.get("hex_colors") or data.get("color_palette")
        self.color_palette = self._process_color_palette(palette_rgb, palette_hex)

    def get_indices(self, indices):
        """
        Maps original index positions to full dataset index positions. For example,
        if idx=100 is in indices, this function returns all the index values of
        elements in data['ids'] that contain 100, so that we're selecting original samples
        with their augmented equivalents.

        :param indices: ndarray of original indices values
        :return: ndarray of dataset index values
        """
        if self.ids is None:
            return

        indices = [i for i, element in enumerate(self.ids) if element in indices]
        return np.asarray(indices)

    def _hex_to_rgb(self, hex_string):
        """
        Take a hex string and convert to a RGB tuple.

        :param hex_string: string of the format #XXXXXX
        :return: (R, G, B) integers
        """
        return tuple(int(hex_string[i: i + 2], 16) for i in (1, 3, 5))

    def _process_color_palette(self, palette_rgb, palette_hex):
        """
        Transform raw data input whether it's a list of RGB tuples or hex strings to a clean RGB
        representation.

        :param palette_rgb: list of RGB tuples
        :param palette_hex: list of hex strings
        :return: ndarray of RGB tuples in np.uint8 format
        """
        if palette_rgb is not None:
            return torch.tensor(palette_rgb, dtype=torch.uint8)

        palette = np.asarray([self._hex_to_rgb(color) for color in palette_hex], dtype=np.uint8)
        return torch.from_numpy(palette)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Convert dataset to torch tensors for both image and embeddings for the model

        :param idx: idx being retrieved
        :return: (image, embedding) tensors
        """
        return super().__getitem__(idx)


def load_data(path, batch_size, train_size=0.9, num_workers=0, drop_last=False, seed=0):
    """
    Load data from raw data path.

    :param path: the path to the npy pickle of data
    :param batch_size: number of samples per batch
    :param train_size: proportion of the dataset to include in the train set
    :return: (train_set, test_set, color_palette)
    """
    # load the dataset from raw file
    dataset = ImageDataset(path)

    if dataset.ids is None:
        n_train = int(train_size * len(dataset))
        n_test = len(dataset) - n_train
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    else:
        unique_ids = np.unique(dataset.ids)
        ids_train, ids_test = train_test_split(unique_ids, train_size=train_size, random_state=seed)
        ids_train, ids_test = dataset.get_indices(ids_train), dataset.get_indices(ids_test)
        train_dataset, test_dataset = Subset(dataset, ids_train), Subset(dataset, ids_test)

    # use torch dataloader to simplify batch generation
    # NB: pin_memory=True can help with GPU performance by copying from GPU to CPU pinned memory
    # we should shuffle as the augmentations should be randomly mixed in
    train_set = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    test_set = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )

    return train_set, test_set, dataset.color_palette
