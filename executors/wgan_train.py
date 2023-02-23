import os
import pathlib

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
                    batch_size=64, lr=1e-3, n_critic=4,
                    shuffle=True, debug=True, show_each=200, overfit=False, seed=23)

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

    dataset = MVTEC_AD(dataset_type=DatasetType.TRAIN, data_type=MVTEC_AD.DataClass.Pill, data_path=cfg.DATASET_DIR,
                       transform=image_transforms, mask_transform=mask_transforms)

    dataloaders = {
        train_key: DataLoader(dataset=dataset, batch_size=cfg.batch_size,
                              num_workers=1, pin_memory=True,
                              drop_last=True, shuffle=True)}

    generator = Generator(in_depth=cfg.z_depth, end_depth=32, out_depth=3).to(cfg.device)
    generator.apply(weights_init)
    generator_optimizer = Adam(generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    generator_criterion = generator_loss

    discriminator = Discriminator(in_depth=3, start_depth=32).to(cfg.device)
    discriminator.apply(weights_init)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)
    discriminator_criterion = WassersteinGradientPenaltyLoss(discriminator, cfg.penalty, device=cfg.device)

    fixed_noise = None

    epoch_manager = EpochManagerAnoGAN(generator=generator, generator_optimizer=generator_optimizer,
                                       generator_criterion=generator_criterion,
                                       discriminator=discriminator, discriminator_optimizer=discriminator_optimizer,
                                       discriminator_criterion=discriminator_criterion,
                                       fixed_noise=fixed_noise,
                                       dataloaders_dict=dataloaders, cfg=cfg)

    final_train = True
    if final_train:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
    else:
        torch.autograd.set_detect_anomaly(True)

    epochs = 10_000
    epoch = 1
    save_each = 10
    # epoch_manager.load(path=..., noise=False)
    writer = SummaryWriter(log_dir=cfg.LOG_PATH)
    tr_str, ev_str = 'train', 'eval'
    experiment_name = 'exp1'

    # train GAN
    for epoch in range(epoch, epochs + epoch):
        EPOCH = f'EPOCH #{epoch}'
        print(f'{EPOCH:_^20}')
        print(f'{tr_str:_^10}')
        with Timer('Train'):
            epoch_manager.train_gan(train_key)

        print(f'{ev_str:_^10}')
        with Timer('Evaluation'):
            epoch_manager.save_generate_images(os.path.join(cfg.ROOT_DIR, 'imgs'), epoch, nrow=16)

        if epoch % save_each == 0:
            epoch_manager.save_model(epoch, os.path.join(cfg.ROOT_DIR, 'saves', experiment_name))

    stage = train_key
    d_losses, g_losses = epoch_manager.discriminator_losses[stage], epoch_manager.generator_losses[stage]
    f_accs, r_accs = epoch_manager.fake_accuracy[stage], epoch_manager.real_accuracy[stage]
    d_losses = np.cumsum(d_losses) / np.arange(1, 1 + len(d_losses))
    g_losses = np.cumsum(g_losses) / np.arange(1, 1 + len(g_losses))
    f_accs = np.cumsum(f_accs) / np.arange(1, 1 + len(f_accs))
    r_accs = np.cumsum(r_accs) / np.arange(1, 1 + len(r_accs))
    for i, (dl, gl, fa, ra) in enumerate(zip(d_losses, g_losses, f_accs, r_accs)):
        writer.add_scalar(f'{stage}/D_Loss', dl, i)
        writer.add_scalar(f'{stage}/G_Loss', gl, i)
        writer.add_scalar(f'{stage}/Fake Accuracy', fa, i)
        writer.add_scalar(f'{stage}/Real Accuracy', ra, i)


if __name__ == '__main__':
    main()
