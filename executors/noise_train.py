import os
import pathlib

import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss, Parameter
from torch.optim import RMSprop
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
    from models import Generator
    from utils import Timer
    from transforms import UnNormalize
    from datasets import DatasetType, MVTEC_AD

    DATA_PATH = pathlib.Path(r'D:\datasets\pill')

    cfg = GANConfig(model_name='AnoWGAN', z_depth=100, device='cuda',
                    dataset_name='MVTEC', DATASET_DIR=DATA_PATH,
                    batch_size=167, lr=1e-2, z_train_steps=5_000,
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
        lambda x: x.round(),
    ])

    un_norm = UnNormalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))

    dataset = MVTEC_AD(dataset_type=DatasetType.TEST, data_type=MVTEC_AD.DataClass.Pill, data_path=cfg.DATASET_DIR,
                       transform=image_transforms, mask_transform=mask_transforms)  # torch mvtec

    dataloaders = {valid_key: DataLoader(dataset=dataset, batch_size=cfg.batch_size,
                                         num_workers=1, pin_memory=True,
                                         drop_last=False, shuffle=False)}

    generator = Generator(in_depth=cfg.z_depth, end_depth=32, out_depth=3).to(cfg.device)

    epoch_manager = EpochManagerAnoGAN(generator=generator,
                                       dataloaders_dict=dataloaders, cfg=cfg)
    noise = Parameter(torch.FloatTensor(size=(cfg.batch_size, cfg.z_depth, 1, 1)).to(cfg.device), requires_grad=True)
    noise_optimizer = RMSprop((noise,), lr=cfg.lr)
    noise_criterion = MSELoss(reduction='sum')

    final_train = True
    if final_train:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
    else:
        torch.autograd.set_detect_anomaly(True)

    epochs = 10_000
    epoch = 1
    save_each = 100
    # epoch_manager.load_model(path=..., noise=True)
    epoch_manager.set_trainable_noise(noise, noise_optimizer, noise_criterion)
    writer = SummaryWriter(log_dir=cfg.LOG_PATH)
    tr_str, ev_str = 'train', 'eval'
    experiment_name = 'exp1'

    # train Noise
    with Timer('Train'):
        epoch_manager.train_noise_vector(valid_key)

    epoch_manager.save_model(epoch, os.path.join(cfg.ROOT_DIR, 'saves', experiment_name))

    losses = epoch_manager.noise_losses
    write_metric(writer, f'{valid_key}/Real Accuracy', losses)


def write_metric(writer, tag, metric):
    metric = np.cumsum(metric) / np.arange(1, 1 + len(metric))
    for i, m in enumerate(metric):
        writer.add_scalar(tag, m, i)


if __name__ == '__main__':
    main()
