import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from torch_model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device...")

class imageDataSet(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx]
    
def load_data(path, scaling_factor=6):
    data = np.load(path, allow_pickle=True).item()
    images = np.array(data['images'])
    labels = data['labels']

    embeddings = data['embeddings']
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    embeddings = embeddings * scaling_factor

    images, images_test, labels, labels_test, embeddings, embeddings_test = train_test_split(
    images, labels, embeddings, test_size=24, random_state=seed)

    train_dataset = [embeddings, images]
    test_dataset = [embeddings_test, images_test]

    train_set = DataLoader(imageDataSet(train_dataset),
                       batch_size=BATCH_SIZE,
                       shuffle=True,
                       num_workers= 8 if device == 'cuda' else 1,
                       pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

    test_set = DataLoader(imageDataSet(test_dataset),
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers= 8 if device == 'cuda' else 1,
                      pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

    return train_set, test_set

def train(model, EPOCHS, batch_size, sample_interval=10):

    train_set, test_set = load_data("maps_gpt4_aug.npy")

    loss_metric_train = torch.zeros(EPOCHS).to(device)

    model.to(device)

    optimizer = optim.Adam(model.parameters())

    for epoch in range(EPOCHS):

        for embeddings, ytrue in train_set:

            optimizer.zero_grad()
            outputs = model(emb.to(device), torch.rand(len(emb), 5).to(device))
            loss = nn.NLLLoss()(torch.log(outputs), ytrue.argmax(dim=1).to(device))

            loss_metric_train[epoch] += loss

            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    input_shape = (10, 10, 16)
    epochs = 100
    batch_size = 256
    encpic = Generator(
        device,
        noise_emb_size=input_shape,
        text_emb_size=384,
        num_filters=128,
        num_residual_blocks=3,
        kernel_size=7,
        conv_size=4
        )
    train(encpic, epochs, batch_size, sample_interval=10)