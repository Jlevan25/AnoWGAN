from collections import deque

import torch
from torch import autograd
from torch.autograd import Variable
from torch.nn import Module


class WassersteinGradientPenaltyLoss(Module):
    def __init__(self, discriminator, penalty, is_batched=True, device='cpu'):
        super().__init__()
        self._discriminator = discriminator
        self.penalty = penalty
        self.device = device
        self.is_batched = is_batched

    def forward(self, real, fake, real_logits, fake_logits):
        shape = real_logits.shape
        weight = torch.rand(shape) if self.is_batched else torch.rand(1).item()
        inter_input = torch.lerp(real, fake, weight.to(self.device))
        inter_output, inter_logits = self._discriminator(inter_input)

        gradients = autograd.grad(
            outputs=inter_logits,
            inputs=inter_input,
            grad_outputs=Variable(torch.ones(shape), requires_grad=True).to(self.device),  # TODO cuda?
            create_graph=True,
            retain_graph=True,  # TODO retain_graph?
            only_inputs=True,
        )[0]
        gradients = gradients.flatten(1).norm(2, dim=-1)
        return torch.mean(fake_logits - real_logits) + self.penalty * torch.mean((gradients - 1) ** 2)  # TODO mean?


def generator_loss(fake_logits):
    return -fake_logits.mean()


if __name__ == '__main__':
    from utils import Timer

    with Timer('muladd'):
        first = torch.ones(10_000, 100_000)
        # print(first*0+1)
    with Timer('pow'):
        second = torch.zeros(10_000, 100_000)
        print(second ** 0)
    print()
