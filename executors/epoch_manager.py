import os.path
from typing import Iterator, Union, List

import Levenshtein as Levenshtein
import torch
import torchvision.utils
from torch import tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader


# from datasets import CocoLocalizationDataset


class EpochManagerAnoGAN:
    def __init__(self,
                 cfg,
                 generator=None,
                 generator_optimizer=None,
                 generator_criterion=None,
                 discriminator=None,
                 discriminator_optimizer=None,
                 discriminator_criterion=None,
                 dataloaders_dict=None,
                 device=None,
                 fixed_noise=None):
        self.cfg = cfg

        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_criterion = generator_criterion
        self.generator_losses = dict()
        self.fake_accuracy = dict()

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_criterion = discriminator_criterion
        self.discriminator_losses = dict()
        self.real_accuracy = dict()

        self.fixed_noise = fixed_noise is not None
        self.device = self.cfg.device if device is None else device
        self.noise = fixed_noise if self.fixed_noise \
            else torch.FloatTensor(cfg.batch_size, cfg.z_depth, 1, 1).to(self.device)
        self.dataloaders = dataloaders_dict if dataloaders_dict is not None else dict()
        self.valid_noise = None

        self.noise_trainable = None
        self.noise_optimizer = None
        self.noise_criterion = None
        self.noise_losses = []
        self.noises_saved = []

        self._global_step = dict()

    def set_generator(self, generator, optimizer, criterion):
        self.generator = generator
        self.generator_optimizer = optimizer
        self.generator_criterion = criterion

    def set_discriminator(self, discriminator, optimizer, criterion):
        self.discriminator = discriminator
        self.discriminator_optimizer = optimizer
        self.discriminator_criterion = criterion

    def set_trainable_noise(self, noise, optimizer, criterion):
        self.noise_trainable = noise
        self.noise_optimizer = optimizer
        self.noise_criterion = criterion

    def train_gan(self, stage_key):
        self.discriminator.train()
        self.generator.train()
        if stage_key not in self._global_step:
            self._get_global_step(stage_key)

        g_loss = torch.zeros(1)
        for i, (real, targets) in enumerate(self.dataloaders[stage_key], start=1):
            self._global_step[stage_key] += 1
            if not self.fixed_noise:
                self.noise.normal_()
            fake = self.generator(self.noise)
            real = real.to(self.device)
            real_output, real_logits = self.discriminator(real)
            fake_output, fake_logits = self.discriminator(fake)

            d_loss = self.discriminator_criterion(real, fake, real_logits, fake_logits)
            real_acc, fake_acc = ((o.detach().cpu().round() == i).sum().item() / self.cfg.batch_size
                                  for i, o in enumerate((fake_output, real_output)))
            self.discriminator_losses[stage_key].append(d_loss.item())

            self.real_accuracy[stage_key].append(real_acc)
            self.fake_accuracy[stage_key].append(fake_acc)
            for param in self.discriminator.parameters():
                param.grad = None
            d_loss.backward()
            self.discriminator_optimizer.step()

            if i % self.cfg.n_critic == 0:
                if not self.fixed_noise:
                    self.noise.normal_()
                fake = self.generator(self.noise)
                fake_output, fake_logits = self.discriminator(fake)
                g_loss = self.generator_criterion(fake_logits)
                for param in self.generator.parameters():
                    param.grad = None
                g_loss.backward()
                self.generator_optimizer.step()

                self.generator_losses[stage_key].append(g_loss.item())

            if self.cfg.debug and i % self.cfg.show_each == 0 or i == len(self.dataloaders[stage_key]):
                print(f'step : {i}/{len(self.dataloaders[stage_key])}', f'd_loss: {d_loss.item():.4}',
                      f'g_loss: {g_loss.item():.4}', f'real_acc:{real_acc:.4}', f'fake_acc:{fake_acc:.4}',
                      sep='\t\t')

    def train_noise_vector(self, stage_key, save_vectors=True):
        self.generator.eval()
        if self.noise_trainable is None:
            raise ValueError('noise is None')

        for idx, (real, targets, masks) in enumerate(self.dataloaders[stage_key]):
            self.noise_trainable.data.normal_()
            mean_losses = 0
            for i in range(1, 1 + self.cfg.z_train_steps):
                fake = self.generator(self.noise_trainable)

                loss = self.noise_criterion(real.to(self.device), fake)
                self.noise_losses.append(loss.item())
                self.noise_trainable.grad = None
                for param in self.generator.parameters():
                    param.grad = None
                loss.backward()
                self.noise_optimizer.step()
                mean_losses += loss.item()

                if self.cfg.debug and i % self.cfg.show_each == 0 or i == self.cfg.z_train_steps:

                    torchvision.utils.save_image(fake,
                                                 fr'C:\AD\anogan-Jlevan25\imgs\anomaly3\{i}.png',
                                                 nrow=32)
                    print(f'step : {i}/{self.cfg.z_train_steps}', f'loss: {mean_losses / i:.4}', sep='\t\t')

            self.noises_saved.append(self.noise_trainable.data.cpu().detach())

            with torch.no_grad():
                fake = self.generator(self.noise_trainable)
                real, fake = real.cpu(), fake.cpu()
                mq = torch.abs(real-fake)
                masks = masks.repeat(1, 3, 1, 1)
                for t in set(targets.tolist()):
                    mask = targets == t
                    img = torch.cat((fake[mask], real[mask], mq[mask], masks[mask]), 0)
                    name = self.dataloaders[stage_key].dataset.class2name(t)
                    torchvision.utils.save_image(img,
                                                 fr'C:\AD\anogan-Jlevan25\imgs\anomaly2\{name}.png',
                                                 nrow=len(img)//4)

    @torch.no_grad()
    def save_generate_images(self, path, epoch, num_images=None, transforms=None, nrow=8):
        self.discriminator.eval()
        self.generator.eval()

        if not self.fixed_noise:
            if self.valid_noise is None:
                bs = num_images if num_images is not None else self.cfg.batch_size
                self.valid_noise = torch.normal(0, 1, (bs, self.cfg.z_depth, 1, 1))
            noise = self.valid_noise.to(self.device)
        else:
            noise = self.noise

        if num_images is not None:
            noise = noise[:num_images]

        generated = self.generator(noise).cpu()
        if transforms is not None:
            generated = transforms(generated)

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(generated, os.path.join(path, f'{str(epoch)}.png'), nrow=nrow)

    @torch.no_grad()
    def inference(self, stage_key, metrics):
        self.discriminator.eval()
        self.generator.eval()
        out = [[]for _ in range(len(metrics))]
        for i, (real, targets, mask) in enumerate(self.dataloaders[stage_key], start=1):
            targets = (targets > 0).int()
            fake = self.generator(self.noise_trainable.data.to(self.device))
            real_output, real_logits = self.discriminator(real.to(self.device))
            fake_output, fake_logits = self.discriminator(fake)

            r_score = torch.abs(real - fake.cpu()).sum((1, 2, 3))
            d_score = torch.abs(real_logits.cpu() - fake_logits.cpu()).sum((1, 2, 3))
            ano_score = torch.lerp(r_score, d_score, 0.1)
            ano_score_n = (ano_score - ano_score.min()) / (ano_score.max() - ano_score.min())
            for j, metric in enumerate(metrics):
                out[j].append(metric(ano_score_n, targets))

            return out

        print()

    def save_model(self, epoch, path=None, noise=False):
        path = self.cfg.SAVE_PATH if path is None else path
        path = os.path.join(path, f'{epoch}')
        if not os.path.exists(path):
            os.makedirs(path)

        mod_path = os.path.join(path, 'model.pth')
        met_path = os.path.join(path, 'metrics.pth')

        if noise:
            mod_checkpoint = dict(epoch=self._global_step,
                                  generator=self.generator.state_dict(),
                                  noises=self.noises_saved,
                                  noise_trainable=self.noise_trainable,
                                  noises_optimizer=self.noise_optimizer,
                                  )
            met_checkpoint = dict(epoch=self._global_step,
                                  noises=self.noise_losses,
                                  )
        else:
            mod_checkpoint = dict(epoch=self._global_step,
                                  discriminator=self.discriminator.state_dict(),
                                  discriminator_optimizer=self.discriminator_optimizer.state_dict(),
                                  generator=self.generator.state_dict(),
                                  generator_optimizer=self.generator_optimizer.state_dict(),
                                  )
            met_checkpoint = dict(epoch=self._global_step,
                                  generator_losses=self.generator_losses,
                                  discriminator_losses=self.discriminator_losses,
                                  fake_accuracy=self.fake_accuracy,
                                  real_accuracy=self.real_accuracy,
                                  noise=self.noise
                                  )
        torch.save(mod_checkpoint, mod_path)
        torch.save(met_checkpoint, met_path)
        print('model saved, epoch:', epoch)

    def load_model(self, path, noise):
        checkpoint = torch.load(os.path.join(path, 'model.pth'), map_location=torch.device(self.device))
        self._global_step = checkpoint['epoch']
        if noise:
            self.noise_trainable = checkpoint['noise_trainable']
            # self.noises_saved = checkpoint['noises_saved']
        else:
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.generator.load_state_dict(checkpoint['generator'])

    def load_metrics(self, path, noise):
        checkpoint = torch.load(os.path.join(path, 'metrics.pth'))
        if noise:
            self.noise_losses = checkpoint['noise_losses']
        else:
            self.generator_losses = checkpoint['generator_losses']
            self.discriminator_losses = checkpoint['discriminator_losses']
            self.fake_accuracy = checkpoint['fake_accuracy']
            self.real_accuracy = checkpoint['real_accuracy']
            # self.noise = checkpoint['noise']
        print('model loaded')

    def load(self, path, noise):
        self.load_model(path, noise)
        self.load_metrics(path, noise)

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1
        self.discriminator_losses[data_type] = []
        self.generator_losses[data_type] = []
        self.fake_accuracy[data_type] = []
        self.real_accuracy[data_type] = []
