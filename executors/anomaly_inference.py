import os
import pathlib

import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss, Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryAveragePrecision, BinaryConfusionMatrix
import sys

from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    from configs import GANConfig
    from executors.epoch_manager import EpochManagerAnoGAN
    from losses import WassersteinGradientPenaltyLoss, generator_loss
    from models import Generator, Discriminator
    from utils import Timer
    from transforms import UnNormalize
    from datasets import DatasetType, MVTEC_AD

    DATA_PATH = pathlib.Path(r'D:\datasets\pill')

    cfg = GANConfig(model_name='AnoWGAN', z_depth=100, device='cuda',
                    dataset_name='MVTEC', DATASET_DIR=DATA_PATH,
                    batch_size=167, lr=1e-3, z_train_steps=10_000,
                    shuffle=True, debug=True, show_each=100, overfit=False, seed=23)

    # keys = train_key, valid_key, test_key = 'train', 'valid', 'test'
    keys = train_key, valid_key = 'train', 'valid'

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    base_transfrom = [transforms.Resize(128),
                      transforms.ToTensor()]

    image_transforms = transforms.Compose([
        *base_transfrom,
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
    ])

    mask_transforms = transforms.Compose([
        *base_transfrom,
        lambda x: (x > 127).float(),
    ])

    un_norm = UnNormalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))

    dataset = MVTEC_AD(dataset_type=DatasetType.TEST, data_type=MVTEC_AD.DataClass.Pill, data_path=cfg.DATASET_DIR,
                       transform=image_transforms, mask_transform=mask_transforms)  # torch mvtec

    dataloaders = {valid_key: DataLoader(dataset=dataset, batch_size=cfg.batch_size,
                                         num_workers=1, pin_memory=True,
                                         drop_last=False, shuffle=False)}

    generator = Generator(in_depth=cfg.z_depth, end_depth=32, out_depth=3).to(cfg.device)
    generator_optimizer = Adam(generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    generator_criterion = generator_loss

    discriminator = Discriminator(in_depth=3, start_depth=32).to(cfg.device)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)
    discriminator_criterion = WassersteinGradientPenaltyLoss(discriminator, cfg.penalty, device=cfg.device)

    epoch_manager = EpochManagerAnoGAN(generator=generator, generator_optimizer=generator_optimizer,
                                       generator_criterion=generator_criterion,
                                       discriminator=discriminator, discriminator_optimizer=discriminator_optimizer,
                                       discriminator_criterion=discriminator_criterion,
                                       dataloaders_dict=dataloaders, cfg=cfg)

    noise = Parameter(torch.FloatTensor(size=(cfg.batch_size, cfg.z_depth, 1, 1)).to(cfg.device), requires_grad=True)
    noise_optimizer = Adam((noise,), lr=cfg.lr, betas=cfg.betas)
    noise_criterion = MSELoss()

    final_train = True
    if final_train:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
    else:
        torch.autograd.set_detect_anomaly(True)

    # epoch_manager.load_model(path=..., noise=False)
    epoch_manager.set_trainable_noise(noise, noise_optimizer, noise_criterion)
    # epoch_manager.load_model(path=..., noise=True)
    writer = SummaryWriter(log_dir=cfg.LOG_PATH)
    metrics = [BinaryAveragePrecision(), BinaryF1Score(), BinaryConfusionMatrix()]
    with Timer('Inference'):
        ap, f1, cm = epoch_manager.inference(valid_key, metrics)


if __name__ == '__main__':
    main()
