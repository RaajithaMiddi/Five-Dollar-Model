import torch
from torch import nn, optim

from torch_model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device...")

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