"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson@hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from phosphene_encoder import Phoscoder
from simulator_model import Simulator
import scipy.io # for loading .mat file
# horse -> original
# zebra -> target/phosphene

def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, simu, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, grid
):
    H_reals = 0 # H_real is the output of discriminator for real horse image
    H_fakes = 0 # H_fake is the output of discriminator for fake horse image
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse)
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra_endcoding = gen_Z(horse).to(config.DEVICE)
            # simu = Simulator(imgsize=512)
            # fake_zebra = simu(torch.sigmoid(fake_zebra_endcoding), grid, config.DEVICE)

            # simu = Simulator(grid / 1080 * 512, config.DEVICE, imgsize=512)
            fake_zebra = simu(torch.sigmoid(fake_zebra_endcoding))
            # fake_zebra = fake_zebra[None, None] # add batch and channel dimension

            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra)
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra_encoding = gen_Z(fake_horse).to(config.DEVICE)
            # simu_cyc = Simulator(imgsize=512)
            # cycle_zebra = simu_cyc(torch.sigmoid(cycle_zebra_encoding), grid, config.DEVICE)

            # simu = Simulator(grid / 1080 * 512, config.DEVICE, imgsize=512)
            cycle_zebra = simu(torch.sigmoid(cycle_zebra_encoding))
            # cycle_zebra = cycle_zebra[None, None]

            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # identity_zebra_encoding = gen_Z(zebra)
            # simu_id = Simulator(identity_zebra_encoding, grid, imgsize=512)
            # identity_zebra = torch.from_numpy(simu_id.image).to(config.DEVICE)
            # cycle_zebra = cycle_zebra[None, None].float()
            #
            # identity_horse = gen_H(horse)
            # identity_zebra_loss = l1(zebra, identity_zebra)
            # identity_horse_loss = l1(horse, identity_horse)

            # BCE loss for the encoding and pixel sampled from images.
            x, y = grid[:, 0], grid[:, 1]
            # filter = (x >= 0) & (x <= 512) & (y >= 0) & (y <= 512)
            # x, y = x[filter], y[filter]
            criterion = nn.BCEWithLogitsLoss()

            # zebra_pix_sample = horse[:, 0, x, y].to(config.DEVICE)
            # zebra_pix_sample = torch.squeeze(zebra_pix_sample)
            # zebra_pix_sample = (zebra_pix_sample-min(zebra_pix_sample))/(max(zebra_pix_sample)-min(zebra_pix_sample))
            # BCE_loss_zebra = criterion(fake_zebra_endcoding, zebra_pix_sample)

            # cycle_zebra_pix_sample = cycle_horse[:, 0, x, y].to(config.DEVICE)
            # cycle_zebra_pix_sample = torch.squeeze(cycle_zebra_pix_sample)
            # cycle_zebra_pix_sample = (cycle_zebra_pix_sample-min(cycle_zebra_pix_sample))/(max(cycle_zebra_pix_sample)-min(cycle_zebra_pix_sample))
            # BCE_loss_cycle = criterion(cycle_zebra_encoding, cycle_zebra_pix_sample)



            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                # + BCE_loss_zebra
                # + BCE_loss_cycle
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                # + identity_horse_loss * config.LAMBDA_IDENTITY
                # + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()

        if (idx+1) % 200 == 0:
            save_image(fake_horse, f"saved_images/g_original_{idx}.png") # original image
            save_image(fake_zebra, f"saved_images/g_phosphene_{idx}.png") # phosphene image
            # save_image(zebra, f"saved_images/phosphene_{idx}.png") # phosphene image
            # save_image(horse, f"saved_images/cycle_{idx}.png") # cycle image

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    phos_map_mat = scipy.io.loadmat('data/grids/map_1200k.mat')
    grid = phos_map_mat['grid']
    grid = grid/1080*511 # might be 1080 for retinotopic map
    disc_H = Discriminator(in_channels=1).to(config.DEVICE) # classify image of horses
    disc_Z = Discriminator(in_channels=1).to(config.DEVICE) # classify image of zebras
    gen_Z = Phoscoder(img_channels=1, num_residuals=4).to(config.DEVICE) # generate phosphene image (zebra) from orignial (horse)
    gen_H = Generator(img_channels=1, num_residuals=4).to(config.DEVICE) # generate image of horse from zebra
    simu = Simulator(grid / 1080 * 512, config.DEVICE, imgsize=512) # simulate phosphene image from phosphene encoding
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    ) # Adam optimizer for discriminator

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    ) # Adam optimizer for generator

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/original_img",
        root_zebra=config.TRAIN_DIR + "/phosphene_img",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "/original_img",
        root_zebra=config.VAL_DIR + "/phosphene_img",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            simu,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            grid,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()