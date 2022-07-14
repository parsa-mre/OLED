import os
import gc
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
import torch
import config
import models
from tensorflow import keras


def threshold(x):
    a = x.reshape(x.shape[0], -1)
    q = torch.quantile(a, config.T, dim=1).reshape(-1, 1)
    r = F.relu(-a + q)
    r = torch.div(r, (r + .0000000001))
    r = r.reshape(x.shape)
    return r


def single_step(MM, R, dataloader, opt_MM, opt_R, epoch):
    loop = tqdm(dataloader, leave=True)
    for x in loop:
        x = x.to(config.DEVICE)

        # train reconstructor
        reconstructor_normal_output = R(x)
        activation_maps = MM(x)
        activation_maps = activation_maps.detach()
        mask = threshold(activation_maps)
        masked_images = torch.mul(x, mask)

        masked_reconstruction = R(masked_images)
        mask_inverse = (mask == 0).type(torch.float32)
        missing_parts = torch.mul(x, mask_inverse)
        generated_missing_parts = torch.mul(
            masked_reconstruction, mask_inverse)

        l_rec = torch.mean(torch.pow(reconstructor_normal_output - x, 2))
        l_cont = 12.5 * \
            torch.mean(torch.abs(missing_parts - generated_missing_parts))

        reconstructor_loss = l_rec + l_cont
        R.zero_grad()
        reconstructor_loss.backward()
        opt_R.step()

        # train mask model
        if epoch % 3 == 0:
            activation_maps = MM(x)
            mask = threshold(activation_maps)
            masked_images = torch.mul(x, mask)
            masked_reconstruction = R(masked_images)
            mask_inverse = (mask == 0).type(torch.float32)
            missing_parts = torch.mul(x, mask_inverse)
            generated_missing_parts = torch.mul(
                masked_reconstruction, mask_inverse)

            l_mask = torch.mean(torch.pow(masked_images - x, 2))
            l_cont = 12.5 * \
                torch.mean(torch.abs(missing_parts - generated_missing_parts))

            mask_model_loss = -l_mask - l_cont

            MM.zero_grad()
            mask_model_loss.backward()
            opt_MM.step()


def train_model(inlier, batch_size=64, epochs=400):

    # setting seed
    # torch.manual_seed(config.SEED)
    # random.seed(config.SEED)
    # np.random.seed(config.SEED)

    # models
    mask_module = models.MaskModule()
    reconstructor = models.Reconstructor()

    # optimizers
    opt_mm = optim.Adam(mask_module.parameters(),
                        lr=config.LEARNING_RATE, betas=(0.5, 0.9))
    opt_r = optim.Adam(reconstructor.parameters(),
                       lr=config.LEARNING_RATE, betas=(0.5, 0.9))

    # data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test = x_test / x_test.max()
    x_test.reshape(-1, 1, 28, 28)
    x_test = torch.from_numpy(x_test).float().to(config.DEVICE)

    x_train = x_train / x_train.max()
    x_train = x_train.reshape(-1, 1, 28, 28)
    idx = y_train == inlier
    x_train = x_train[idx]
    x_train = torch.from_numpy(x_train).float().to(config.DEVICE)

    train_loader = torch.utils.data.DataLoader(
        x_train, batch_size=batch_size, shuffle=True)

    # training loop
    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch}')
        single_step(mask_module, reconstructor,
                    train_loader, opt_mm, opt_r, epoch=epoch)

    return mask_module, reconstructor
