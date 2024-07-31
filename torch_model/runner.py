import os

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils.images import decode_image_batch, image_grid

from load_data import load_data
from torch_model import DollarModel
from visualization import plot_loss_acc


def _train(images, embeddings, model, optimizer, loss_fn):
    model.train()  # set to training mode
    optimizer.zero_grad()  # clear previous gradient

    # get y_hat and y
    outputs_pred = model(images, embeddings)
    outputs_true = images.argmax(dim=-1)

    # compute loss
    loss = loss_fn(outputs_pred, outputs_true)
    loss.backward()  # backwards pass

    # update params
    optimizer.step()

    return loss.item()


def _eval(images, embeddings, model, loss_fn):
    model.eval()

    with torch.no_grad():
        images_pred = model(images, embeddings)
        images_true = images.argmax(dim=-1)
        loss = loss_fn(images_pred, images_true)

    return images_pred, loss.item()


def _print_training_info(batch_num, loss, model, scheduler):
    # concatenate gradients for analysis
    gradients = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]

    for gradient in gradients:
        if torch.isnan(gradient).any():
            raise Exception("NaN gradient detected; shit is fucked!")

    total_norm = torch.cat(gradients).norm().item()

    print(f"\tbatch (train) {batch_num:02.0f}, loss {loss:.2f}, total norm: {total_norm:.2f}")


def _process_batch(batch, device):
    images, embeddings = batch
    return images.to(device), embeddings.to(device)


def _save_results(images_pred, test_images, palette, epoch):
    preds_for_decode = images_pred.permute(0, 2, 3, 1)
    images_pred = decode_image_batch(preds_for_decode, palette)
    images_true = decode_image_batch(test_images, palette)
    results_grid = image_grid([images_true, images_pred])

    results_grid.save(os.path.join("debug_images", f"results_epoch_{epoch}.png"))

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
    loader_train, loader_test, palette = load_data(file_path, batch_size)
    palette = palette.to(device)  # make sure

    # use a model wrapper here; technically not a generator
    model = DollarModel(device).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    loss_train, loss_val = [], []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss_train = 0
        epoch_loss_val = 0

        learning_rate = scheduler.get_last_lr()[0]

        for i, batch in enumerate(loader_train):

            # unpack a batch to image and embedding components
            train_images, train_embeddings = _process_batch(batch, device)

            # TRAIN the model and accumulate loss
            batch_loss_train = _train(train_images, train_embeddings, model, optimizer, loss_fn)
            epoch_loss_train += batch_loss_train

            _print_training_info(i, batch_loss_train, model, scheduler)

        # average loss over the number of batches
        avg_epoch_loss_train = epoch_loss_train / len(loader_train)
        loss_train.append(avg_epoch_loss_train)

        if epoch % eval_every == 0:

            for i, batch in enumerate(loader_test):
                # evaluate performance on out-of-sample batches and accumulate the loss
                test_images, test_embeddings = _process_batch(batch, device)
                images_pred, batch_loss_val = _eval(test_images, test_embeddings, model, loss_fn)

                epoch_loss_val += batch_loss_val
                print(f"\tbatch (val) {i:02.0f}, validation loss {batch_loss_val:.2f}")

                _save_results(images_pred, test_images, palette, epoch)

            # compute the average epoch loss
            avg_epoch_loss_val = epoch_loss_val / len(loader_test)
            loss_val.append(avg_epoch_loss_val)

        tqdm.write(
            f"\nEpoch {epoch + 1}/{epochs}: Learning Rate: {learning_rate:.6f} Avg Training Loss: "
            f"{avg_epoch_loss_train:.6f} Avg. Validation Loss: {avg_epoch_loss_val:.6f}"
        )

        # increment scheduler at the epoch level
        scheduler.step()

    torch.save(model.state_dict(), "five_dollar_model.pt")
    plot_loss_acc(loss_train, loss_val)
