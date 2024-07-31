import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils.images import decode_image_batch, image_grid

from load_data import load_data
from torch_model import Generator
from visualization import plot_loss_acc


class GeneratorModule(nn.Module):
    def __init__(
            self, device: torch.device, generator: Generator, palette: torch.Tensor
    ):
        super().__init__()
        self.device = device
        self.generator = generator.to(device)
        self.loss_fn = torch.nn.NLLLoss()
        self.results_table = None
        self.palette = palette.to(device)

    # have to generate embeddings here as otherwise we cannot run the sentence transformer
    # on the GPU
    def process_batch(self, batch):
        images, embeddings = batch
        embeddings = embeddings.to(self.device)
        images = images.to(self.device)
        return images, embeddings

    def training_step(self, batch):
        self.generator.train()

        images, embeddings = self.process_batch(batch)
        pred_out = self.generator(images, embeddings)
        true_classes = images.argmax(dim=-1)
        # print(f'pred_out shape: {pred_out.shape}')
        # print(f'true_classes shape: {true_classes.shape}')

        loss = self.loss_fn(torch.log(pred_out), true_classes)
        # acc = torch.sum(true_classes == pred_out.max(1)).item() / true_classes.size(0)
        return loss  # , acc

    def eval_step(self, batch, epoch: int):
        self.generator.eval()

        with torch.no_grad():
            images, embeddings = self.process_batch(batch)
            pred_out = self.generator(images, embeddings)
            image_classes = images.argmax(dim=-1)
            loss = self.loss_fn(torch.log(pred_out), image_classes)

        preds_for_decode = pred_out.permute(0, 2, 3, 1)
        pred_images = decode_image_batch(preds_for_decode, self.palette)

        input_images = decode_image_batch(images, self.palette)
        results_grid = image_grid([input_images, pred_images])

        # acc = torch.sum(pred_out == image_classes).item() / len(image_classes)

        results_grid.save(os.path.join("debug_images", f"results_epoch_{epoch}.png"))
        return loss  # , acc


def train(file_path, hyperparameters, device, eval_every=1):
    """
    Run the training loop
    :param file_path: path to data file
    :param hyperparameters: dictionary of hyperparameters
    :param device: torch device
    """

    # unpack hyperparameters for the model to use
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    lr = hyperparameters["learning_rate"]

    # generate batch tensors via TensorDataset and Dataloader
    train_set, test_set, palette = load_data(file_path, batch_size)

    # use a model wrapper here; technically not a generator
    model = GeneratorModule(device, Generator(device), palette)

    loss_train = []
    loss_val = []

    # train_acc = []
    # val_acc = []

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    for epoch in tqdm(range(epochs), desc="Training on epoch"):
        print()

        epoch_loss_train = 0

        for j, batch in enumerate(train_set):
            optimizer.zero_grad()

            batch_loss_train = model.training_step(batch)
            epoch_loss_train += batch_loss_train.item()
            # acc_total += acc

            batch_loss_train.backward()
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            total_norm = torch.cat(grads).norm().item()
            learning_rate: float = scheduler.get_last_lr()[0]

            print(
                "Epoch {:03.0f}, batch {:02.0f}, loss {:.2f}, total norm: {:.2f}, learning rate: {:.8f}".format(
                    epoch, j, batch_loss_train.item(), total_norm, learning_rate
                )
            )
            optimizer.step()
            scheduler.step()

        # should be len training set / batch_size)
        avg_epoch_loss_train = epoch_loss_train / len(train_set)
        loss_train.append(avg_epoch_loss_train)

        if epoch % eval_every == 0:
            epoch_loss_val = 0

            for j, batch in enumerate(test_set):
                batch_loss_val = model.eval_step(batch, epoch)
                epoch_loss_val += batch_loss_val.item()

                # val_acc.append(val_acc)
                print(f"Validation loss: {batch_loss_val.item()}")
            avg_epoch_loss_val = epoch_loss_val / len(test_set)
            loss_val.append(avg_epoch_loss_val)


    torch.save(model.state_dict(), "five_dollar_model.pt")
    plot_loss_acc(loss_train, loss_val)


def _check_cuda_memory(device_id):
    """
    Given a cuda device number, return free memory
    :param device_id: an integer that maps to which cuda gpu
    :return: memory available in bytes
    """
    memory_total = torch.cuda.get_device_properties(device_id).total_memory
    memory_allocated = torch.cuda.memory_allocated(device_id)
    return memory_total - memory_allocated


def select_best_device(device=None):
    """
    Pick the best device in this order: cuda, mps (apple), cpu
    :return: torch device to use
    """

    # override the logic if we pass in a device
    if device:
        return torch.device(device)

    # pick the best gpu, if it's available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        best_gpu = max(range(num_gpus), key=lambda i: _check_cuda_memory(i))
        return torch.device(f"cuda:{best_gpu}")

    # pick apple mps next, if it's available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # return cpu as a last resort
    return torch.device("cpu")


def _create_folders_if_not_exist(folder_path):
    """
    private function to create a folder if it doesn't already exist
    :param folder_path: the folder and path to it
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
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

    hyperparameters = {
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.0005
    }

    train(file_path, hyperparameters, device)