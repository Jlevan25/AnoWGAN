from torch.nn import Module, Sequential, BatchNorm2d, Conv2d, LeakyReLU, Sigmoid


class Discriminator(Module):
    def __init__(self, in_depth, start_depth):
        super(Discriminator, self).__init__()
        self.layers = Sequential(
            # input is (in_depth) x 64 x 64
            Conv2d(in_depth, start_depth, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            # state size. (start_depth) x 32 x 32
            Conv2d(start_depth, start_depth * 2, 4, 2, 1, bias=False),
            BatchNorm2d(start_depth * 2),
            LeakyReLU(0.2, inplace=True),
            # state size. (start_depth*2) x 16 x 16
            Conv2d(start_depth * 2, start_depth * 4, 4, 2, 1, bias=False),
            BatchNorm2d(start_depth * 4),
            LeakyReLU(0.2, inplace=True),
            # state size. (start_depth*4) x 8 x 8
            Conv2d(start_depth * 4, start_depth * 8, 4, 2, 1, bias=False),
            BatchNorm2d(start_depth * 8),
            LeakyReLU(0.2, inplace=True),
            # state size. (start_depth*8) x 4 x 4
            Conv2d(start_depth * 8, start_depth * 16, 4, 2, 1, bias=False),
            BatchNorm2d(start_depth * 16),
            LeakyReLU(0.2, inplace=True),

            Conv2d(start_depth * 16, 1, 4, 1, 0, bias=False),
        )
        self.out_act = Sigmoid()

    def forward(self, input_):
        features = self.layers[:-1](input_)
        x = self.layers[-1](features)
        return self.out_act(x), features
