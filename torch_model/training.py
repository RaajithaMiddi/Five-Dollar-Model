import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image

from torch_model import Generator
from load_data import load_data
from utils import decode_image_batch, image_grid



class GeneratorModule(nn.Module):
    def __init__(
        self,
        device: torch.device,
        generator: Generator,
        palette: Image.Image
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
        loss = self.loss_fn(torch.log(pred_out), true_classes)
        return loss

    def eval_step(self, batch, epoch: int):
        images, embeddings = self.process_batch(batch)
        input_images = decode_image_batch(images, self.palette)

        pred_out = self.generator(images, embeddings)
        preds_for_decode = pred_out.permute(0, 2, 3, 1)
        pred_images = decode_image_batch(preds_for_decode, self.palette)

        results_grid = image_grid([input_images, pred_images])
        image_classes = images.argmax(dim=-1)
        loss = self.loss_fn(torch.log(pred_out), image_classes)
        results_grid.save(os.path.join("debug_images", f"results_epoch_{epoch}.png"))

def train(EPOCHS, batch_size, lr, device):

    train_set, test_set, palette = load_data("datasets/sprite_gpt4aug.npy", batch_size)

    model = GeneratorModule(device, Generator(device), palette)

    loss_metric_train = torch.zeros(EPOCHS).to(device)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    for epoch in tqdm(range(EPOCHS), desc ="Training on epoch"):

        for j, batch in enumerate(train_set):
            optimizer.zero_grad()

            loss = model.training_step(batch)
            loss_metric_train[epoch] += loss

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

        eval_every = 5
        if epoch % eval_every == 0:
            for j, batch in enumerate(test_set):
                print("Running eval..")
                model.eval_step(batch, epoch)
                break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device...")

    epochs = 2 # 100
    batch_size = 256
    lr = 0.0005
    train(epochs, batch_size, lr, device)