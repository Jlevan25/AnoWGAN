from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh


class Generator(Module):
    def __init__(self, in_depth, end_depth, out_depth):
        super(Generator, self).__init__()
        self.layers = Sequential(
            # input is Z, going into a convolution
            ConvTranspose2d(in_depth, end_depth * 16, 4, 1, 0, bias=False),
            BatchNorm2d(end_depth * 16),
            ReLU(True),
            ConvTranspose2d(end_depth * 16, end_depth * 8, 4, 2, 1, bias=False),
            BatchNorm2d(end_depth * 8),
            ReLU(True),
            # state size. (start_depth*8) x 4 x 4
            ConvTranspose2d(end_depth * 8, end_depth * 4, 4, 2, 1, bias=False),
            BatchNorm2d(end_depth * 4),
            ReLU(True),
            # state size. (start_depth*4) x 8 x 8
            ConvTranspose2d(end_depth * 4, end_depth * 2, 4, 2, 1, bias=False),
            BatchNorm2d(end_depth * 2),
            ReLU(True),
            # state size. (start_depth*2) x 16 x 16
            ConvTranspose2d(end_depth * 2, end_depth, 4, 2, 1, bias=False),
            BatchNorm2d(end_depth),
            ReLU(True),
            # state size. (start_depth) x 32 x 32
            ConvTranspose2d(end_depth, out_depth, 4, 2, 1, bias=False),
            Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input_):
        return self.layers(input_)
