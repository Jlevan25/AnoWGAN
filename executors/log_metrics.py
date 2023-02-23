import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
import sys

sys.path.append(r'C:\AdvCV\WasGAN')


def main():
    from configs import GANConfig
    from executors.epoch_manager import EpochManagerAnoGAN
    from losses import WassersteinGradientPenaltyLoss, generator_loss
    from models import Generator, Discriminator

    cfg = GANConfig(model_name='AnoWGAN', z_depth=100, device='cpu',
                    dataset_name='MVTec', DATASET_DIR=r'D:\datasets\mnist',
                    batch_size=4, lr=1e-4, class_idx=8,
                    shuffle=True, debug=True, show_each=200, overfit=False, seed=23)

    # keys = train_key, valid_key, test_key = 'train', 'valid', 'test'
    keys = train_key, valid_key = 'train', 'valid'

    generator = Generator(in_depth=cfg.z_depth, end_depth=32, out_depth=1)
    generator_optimizer = Adam(generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    generator_criterion = generator_loss

    discriminator = Discriminator(in_depth=1, start_depth=32)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)
    discriminator_criterion = WassersteinGradientPenaltyLoss(discriminator, cfg.penalty, device=cfg.device)

    epoch_manager = EpochManagerAnoGAN(generator, generator_optimizer, generator_criterion,
                                       discriminator, discriminator_optimizer, discriminator_criterion,
                                       cfg=cfg)

    epoch_manager.load_metrics(r'C:\AD\anogan-Jlevan25\saves\exp1\4300', False)
    writer = SummaryWriter(log_dir=cfg.LOG_PATH)

    stage = train_key
    d_losses, g_losses = epoch_manager.discriminator_losses[stage], epoch_manager.generator_losses[stage]
    f_accs, r_accs = epoch_manager.fake_accuracy[stage], epoch_manager.real_accuracy[stage]
    write_metric(writer, f'{stage}/D_Loss', d_losses)
    write_metric(writer, f'{stage}/G_Loss', g_losses)
    write_metric(writer, f'{stage}/Fake Accuracy', f_accs)
    write_metric(writer, f'{stage}/Real Accuracy', r_accs)


def write_metric(writer, tag, metric):
    metric = np.cumsum(metric) / np.arange(1, 1 + len(metric))
    for i, m in enumerate(metric):
        writer.add_scalar(tag, m, i)


if __name__ == '__main__':
    main()
