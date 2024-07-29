import os

import torch
from load_data import load_data
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from visualization import plot_loss_acc

from torch_model import Generator
from utils import decode_image_batch, image_grid


class GeneratorModule(nn.Module):
    def __init__(
        self, device: torch.device, generator: Generator, palette: Image.Image
    ):
        super().__init__()
        self.device = device
        self.generator = generator.to(device)
        self.loss_fn = torch.nn.NLLLoss()
        self.results_table = None
        self.palette = palette

    # have to generate embeddings here as otherwise we cannot run the sentence transformer
    # on the GPU
    def process_batch(self, batch):
        images, captions, embeddings = batch
        embeddings = embeddings.to(self.device)
        images = images.to(self.device)
        return images, embeddings

    def training_step(self, batch):
        images, embeddings = self.process_batch(batch)
        pred_out = self.generator(images, embeddings)
        true_classes = images.argmax(dim=-1)
        loss = self.loss_fn(torch.log(pred_out), true_classes.squeeze(1))
        # acc = torch.sum(true_classes == pred_out.max(1)).item() / true_classes.size(0)
        return loss  # , acc

    def eval_step(self, batch, epoch: int):
        images, embeddings = self.process_batch(batch)
        input_images = decode_image_batch(images, self.palette)

        pred_out = self.generator(images, embeddings)
        preds_for_decode = pred_out.permute(0, 2, 3, 1)
        pred_images = decode_image_batch(preds_for_decode, self.palette)

        results_grid = image_grid([input_images, pred_images])
        image_classes = images.argmax(dim=-1)
        loss = self.loss_fn(torch.log(pred_out), image_classes)
        # acc = torch.sum(pred_out == image_classes).item() / len(image_classes)
        if os.path.exists("debug_images") is False:
            os.makedirs("debug_images")
        results_grid.save(os.path.join("debug_images", f"results_epoch_{epoch}.png"))
        return loss  # , acc


def train(file_path, hyperparameters, device):
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    lr = hyperparameters["learning_rate"]

    train_set, test_set, palette = load_data(file_path, batch_size)

    model = GeneratorModule(device, Generator(device), palette)

    train_loss = []
    train_acc = []
    eval_loss = []
    val_acc = []

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    for epoch in tqdm(range(epochs), desc="Training on epoch"):

        batch_loss = 0
        acc_total = 0
        for j, batch in enumerate(train_set):
            optimizer.zero_grad()

            loss = model.training_step(batch)
            batch_loss += loss.item()
            # acc_total += acc

            loss.backward()
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            total_norm = torch.cat(grads).norm().item()
            learning_rate: float = scheduler.get_last_lr()[0]

            print(
                "Epoch {:03.0f}, batch {:02.0f}, loss {:.2f}, total norm: {:.2f}, learning rate: {:.8f}".format(
                    epoch, j, loss.item(), total_norm, learning_rate
                )
            )
            optimizer.step()
            scheduler.step()

        train_loss.append(
            batch_loss / len(train_set)
        )  # should be len training set / batch_size)
        # train_acc.append(acc_total / len(train_set))

        eval_every = 1
        if epoch % eval_every == 0:
            for j, batch in enumerate(test_set):
                val_loss = model.eval_step(batch, epoch)
                eval_loss.append(val_loss.item())
                # val_acc.append(val_acc)
                print(f"Validation loss: {val_loss.item()}")
                break

    torch.save(model.state_dict(), "five_dollar_model.pt")
    plot_loss_acc(train_loss, eval_loss)


def _check_cuda_memory(device_id):
    return torch.cuda.get_device_properties(
        device_id
    ).total_memory - torch.cuda.memory_allocated(device_id)


def select_best_device(device=None):
    """
    Pick the best device in this order: cuda, mps (apple), cpu
    :return: torch device to use
    """
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        best_gpu = max(
            range(torch.cuda.device_count()), key=lambda i: _check_cuda_memory(i)
        )
        return torch.device(f"cuda:{best_gpu}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _create_folders_if_not_exist(folder_path):
    """
    private function to create a folder if it doesn't already exist
    :param folder_path: the folder and path to it
    """
    try:
        os.makedirs(folder_path, exist_ok=False)
        print(f"Directory '{folder_path}' created successfully or already exists.")
    except Exception as e:
        print(f"Error creating directory '{folder_path}': {e}")


def create_directories():
    """
    code currently will throw a fit if debug_images folder isn't already populated
    """
    _create_folders_if_not_exist("debug_images")


if __name__ == "__main__":
    device = select_best_device()
    print(f"Using {device} device...")

    create_directories()

    # file_path = "datasets/sprite_gpt4aug.npy"
    file_path = "datasets/data_train.npy"

    hyperparameters = {"epochs": 2, "batch_size": 256, "learning_rate": 0.0005}

    train(file_path, hyperparameters, device)
