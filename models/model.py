import torch
from torch import nn
from enum import IntEnum

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device...")


class ResidualBlock(nn.Module):
    def __init__(
        self, kernel_size: int = 7, num_filters: int = 128, upsampling=False
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
        )
        self.upsampling = upsampling
        if self.upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        if self.upsampling:
            x = self.upsample(x)
        x1 = self.layers(x)
        return x1 + x


class Generator(nn.Module):
    def __init__(
        self,
        device: torch.device,
        noise_emb_size: int = 5,
        text_emb_size: int = 384,
        num_colors: int = 16,
        num_filters: int = 128,
        num_residual_blocks: int = 5,
        kernel_size: int = 7,
        conv_size: int = 4,
    ):
        super().__init__()

        self.noise_emb_size = noise_emb_size
        self.text_emb_size = text_emb_size
        self.num_filters = num_filters
        self.device = device
        self.num_colors = num_colors
        self.conv_kernel_size = conv_size

        self.reshape_layer = nn.Linear(
            noise_emb_size + self.text_emb_size,
            self.conv_kernel_size * self.conv_kernel_size * num_filters,
        )

        self.res_blocks = nn.Sequential()
        for i in range(num_residual_blocks):
            self.res_blocks.append(ResidualBlock(kernel_size, self.filter_count, i < 2))

        out_conv_kernel = 3
        self.out_conv = nn.Conv2d(
            in_channels=self.num_filters, out_channels=16, kernel_size=out_conv_kernel
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, caption_enc):
        batch_size = image.shape[0]
        noise = torch.randn((batch_size, self.noise_emb_size)).to(self.device)
        input_emb = torch.cat([noise, caption_enc], 1).to(self.device)
        x = self.reshape_layer(input_emb)
        x = x.view(-1, self.num_filters, self.conv_kernel_size, self.conv_kernel_size)
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return self.softmax(x)
