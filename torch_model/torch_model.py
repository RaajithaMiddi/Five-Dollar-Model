import torch
from torch import nn

# torch.manual_seed(42)


class ResidualBlock(nn.Module):
    def __init__(
        self, kernel_size: int = 7, num_filters: int = 128, upsampling=True
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.upsampling = upsampling
        self.layers = nn.Sequential(
            nn.Dropout(0.7),  # add dropout for generalization
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Dropout(0.7),  # add dropout for generalization
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            # nn.Dropout(0.7),  # add dropout for generalization
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # self.layer_norm = (nn.LayerNorm([num_filters, 16, 16]),)
        # self.inst_norm = nn.InstanceNorm2d(num_filters, affine=True)

    def forward(self, x):
        if self.upsampling:
            x = self.upsample(x)
        residual = x
        x = self.layers(x)
        # w, b = self.layer_norm(x)
        # x = self.inst_norm(x)
        return residual + x


class DollarModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        noise_emb_size: int = 5,
        num_filters: int = 128,
        num_residual_blocks: int = 5,  # 11 for 500 epochs
        kernel_size: int = 7,
        conv_size: int = 4,
    ):
        super().__init__()

        self.noise_emb_size = noise_emb_size
        self.text_emb_size: int = 384
        self.num_filters = num_filters
        self.device = device
        self.conv_kernel_size = conv_size
        self.num_res_blocks = num_residual_blocks

        self.reshape_layer = nn.Linear(
            noise_emb_size + self.text_emb_size,
            self.conv_kernel_size * self.conv_kernel_size * num_filters,
        )
        torch.nn.init.xavier_uniform_(self.reshape_layer.weight)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(kernel_size, self.num_filters, i < 2)
                for i in range(num_residual_blocks)
            ]
        )
        self.residual_blocks.apply(self.init_weights)
        out_conv_kernel = 17
        self.out_conv = nn.Conv2d(
            in_channels=self.num_filters, out_channels=16, kernel_size=out_conv_kernel
        )
        torch.nn.init.xavier_uniform_(self.out_conv.weight)
        # self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, caption_enc):
        batch_size = image.shape[0]
        noise = torch.randn((batch_size, self.noise_emb_size)).to(self.device)
        input_emb = torch.cat([noise, caption_enc], 1).to(self.device)

        x = self.reshape_layer(input_emb)
        x = x.view(-1, self.num_filters, self.conv_kernel_size, self.conv_kernel_size)
        x = self.residual_blocks(x)
        x = self.array_upsample(x)
        x = self.out_conv(x)

        # x = self.relu(x)
        # return self.softmax(x)
        return x

    def array_upsample(self, inp):
        """
        grabbed from: https://discuss.pytorch.org/t/torch-nn-upsample-layer-slow-in-forwards-pass/70934
        upsample with scale factor of * 2
        """
        out = (
            inp[:, :, :, None, :, None]
            .expand(-1, -1, -1, 2, -1, 2)
            .reshape(inp.size(0), inp.size(1), 2 * inp.size(2), 2 * inp.size(3))
        )
        return out

    def init_weights(self, m):
        """
        Initialize the weights of the model
        grabbed from https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        """
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
