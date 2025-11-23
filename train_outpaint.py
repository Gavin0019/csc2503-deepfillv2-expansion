"""
Training script for outpainting with DeepFillv2.

This script supports:
1. Pure outpainting training (outpaint_only=True)
2. Mixed inpainting + outpainting training (outpaint_only=False)

Usage:
    python train_outpaint.py --config configs/train-outpaint.yaml
"""

import os
import time
import argparse
import torch
import torchvision as tv
import torchvision.transforms as T

import model.losses as gan_losses
import utils.misc as misc
from model.networks import Generator, Discriminator
from utils.data import ImageDataset, OutpaintingDataset


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default="configs/train-outpaint.yaml", help="Path to yaml config file")


def training_loop(generator,
                  discriminator,
                  g_optimizer,
                  d_optimizer,
                  gan_loss_g,
                  gan_loss_d,
                  train_dataloader,
                  last_n_iter,
                  writer,
                  config):

    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')

    losses = {}

    generator.train()
    discriminator.train()

    # Initialize dict for logging
    losses_log = {'d_loss': [],
                  'g_loss': [],
                  'ae_loss': [],
                  'ae_loss1': [],
                  'ae_loss2': [],
                  }

    # Training loop
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)
    time0 = time.time()

    # Get outpainting ratio from config (default 1.0 for pure outpainting)
    outpaint_ratio = getattr(config, 'outpaint_ratio', 1.0)

    for n_iter in range(init_n_iter, config.max_iters):
        # Load batch
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch_data = next(train_iter)

        # Handle different dataset return formats
        if isinstance(batch_data, tuple) and len(batch_data) == 4:
            # OutpaintingDataset returns: (img_full, img_masked, mask, padding)
            batch_real, batch_incomplete_raw, mask, _ = batch_data
            batch_real = batch_real.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            # mask needs extra batch dim if not present
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
        else:
            # Standard ImageDataset - generate mask ourselves
            batch_real = batch_data.to(device, non_blocking=True)

            # Decide: outpainting or inpainting
            if torch.rand(1).item() < outpaint_ratio:
                # Outpainting mask
                mask, _ = misc.outpaint_mask_all_sides(config)
                mask = mask.to(device)
            else:
                # Inpainting mask (original behavior)
                bbox = misc.random_bbox(config)
                regular_mask = misc.bbox2mask(config, bbox).to(device)
                irregular_mask = misc.brush_stroke_mask(config).to(device)
                mask = torch.logical_or(irregular_mask, regular_mask).to(torch.float32)

        # Prepare input for generator
        batch_incomplete = batch_real * (1. - mask)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1].to(device)
        x = torch.cat([batch_incomplete, ones_x, ones_x * mask], axis=1)

        # Generate completed images
        x1, x2 = generator(x, mask)
        batch_predicted = x2

        # Apply mask and complete image
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)

        # D training steps
        batch_real_mask = torch.cat(
            (batch_real, torch.tile(mask, [config.batch_size, 1, 1, 1])), dim=1)
        batch_filled_mask = torch.cat((batch_complete.detach(), torch.tile(
            mask, [config.batch_size, 1, 1, 1])), dim=1)

        batch_real_filled = torch.cat((batch_real_mask, batch_filled_mask))

        d_real_gen = discriminator(batch_real_filled)
        d_real, d_gen = torch.split(d_real_gen, config.batch_size)

        d_loss = gan_loss_d(d_real, d_gen)
        losses['d_loss'] = d_loss

        # Update D parameters
        d_optimizer.zero_grad()
        losses['d_loss'].backward()
        d_optimizer.step()

        # G training steps
        losses['ae_loss1'] = config.l1_loss_alpha * \
            torch.mean((torch.abs(batch_real - x1)))
        losses['ae_loss2'] = config.l1_loss_alpha * \
            torch.mean((torch.abs(batch_real - x2)))
        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

        batch_gen = batch_complete
        batch_gen = torch.cat((batch_gen, torch.tile(
            mask, [config.batch_size, 1, 1, 1])), dim=1)

        d_gen = discriminator(batch_gen)

        g_loss = gan_loss_g(d_gen)
        losses['g_loss'] = g_loss
        losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']

        # Update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()

        # LOGGING
        for k in losses_log.keys():
            losses_log[k].append(losses[k].item())

        if n_iter % config.print_iter == 0:
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()

            for k, loss_log in losses_log.items():
                loss_log_mean = sum(loss_log) / len(loss_log)
                print(f"{k}: {loss_log_mean:.4f}")
                if config.tb_logging:
                    writer.add_scalar(
                        f"losses/{k}", loss_log_mean, global_step=n_iter)
                losses_log[k].clear()

        # Save example image grids to tensorboard
        if config.tb_logging \
                and config.save_imgs_to_tb_iter \
                and n_iter % config.save_imgs_to_tb_iter == 0:
            viz_images = [misc.pt_to_image(batch_complete),
                          misc.pt_to_image(batch_incomplete),
                          misc.pt_to_image(x1),
                          misc.pt_to_image(x2)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                         for images in viz_images]

            writer.add_image(
                "Outpainted", img_grids[0], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Input (masked)", img_grids[1], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Stage 1", img_grids[2], global_step=n_iter, dataformats="CHW")
            writer.add_image(
                "Stage 2", img_grids[3], global_step=n_iter, dataformats="CHW")

        # Save example image grids to disk
        if config.save_imgs_to_disc_iter \
                and n_iter % config.save_imgs_to_disc_iter == 0:
            viz_images = [misc.pt_to_image(batch_real),
                          misc.pt_to_image(batch_incomplete),
                          misc.pt_to_image(batch_complete)]
            img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                         for images in viz_images]
            tv.utils.save_image(img_grids,
                                f"{config.checkpoint_dir}/images/iter_{n_iter}.png",
                                nrow=3)

        # Save state dict snapshot
        if n_iter % config.save_checkpoint_iter == 0 \
                and n_iter > init_n_iter:
            misc.save_states("states.pth",
                             generator, discriminator,
                             g_optimizer, d_optimizer,
                             n_iter, config)

        # Save state dict snapshot backup
        if config.save_cp_backup_iter \
                and n_iter % config.save_cp_backup_iter == 0 \
                and n_iter > init_n_iter:
            misc.save_states(f"states_{n_iter}.pth",
                             generator, discriminator,
                             g_optimizer, d_optimizer,
                             n_iter, config)


def main():
    args = parser.parse_args()
    config = misc.get_config(args.config)

    # Set random seed
    if config.random_seed != False:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        import numpy as np
        np.random.seed(config.random_seed)

    # Make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    # Transforms
    transforms = [T.RandomHorizontalFlip(0.5)] if config.random_horizontal_flip else None

    # Dataloading - choose dataset type based on config
    use_outpaint_dataset = getattr(config, 'use_outpaint_dataset', False)

    if use_outpaint_dataset:
        # Use OutpaintingDataset which handles mask generation internally
        train_dataset = OutpaintingDataset(
            config.dataset_path,
            img_shape=config.img_shapes,
            min_crop_ratio=getattr(config, 'min_crop_ratio', 0.5),
            max_crop_ratio=getattr(config, 'max_crop_ratio', 0.8),
            random_crop=config.random_crop,
            scan_subdirs=config.scan_subdirs,
            transforms=transforms)
    else:
        # Use standard ImageDataset, masks generated in training loop
        train_dataset = ImageDataset(
            config.dataset_path,
            img_shape=config.img_shapes,
            random_crop=config.random_crop,
            scan_subdirs=config.scan_subdirs,
            transforms=transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available()
                          and config.use_cuda_if_available else 'cpu')

    # Construct networks
    cnum_in = config.img_shapes[2]
    generator = Generator(cnum_in=cnum_in + 2, cnum_out=cnum_in, cnum=48, return_flow=False)
    discriminator = Discriminator(cnum_in=cnum_in + 1, cnum=64)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

    # Losses
    if config.gan_loss == 'hinge':
        gan_loss_d, gan_loss_g = gan_losses.hinge_loss_d, gan_losses.hinge_loss_g
    elif config.gan_loss == 'ls':
        gan_loss_d, gan_loss_g = gan_losses.ls_loss_d, gan_losses.ls_loss_g
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    # Resume from existing checkpoint
    last_n_iter = -1
    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore, map_location=device)
        generator.load_state_dict(state_dicts['G'])
        if 'D' in state_dicts.keys():
            discriminator.load_state_dict(state_dicts['D'])
        if 'G_optim' in state_dicts.keys():
            g_optimizer.load_state_dict(state_dicts['G_optim'])
        if 'D_optim' in state_dicts.keys():
            d_optimizer.load_state_dict(state_dicts['D_optim'])
        if 'n_iter' in state_dicts.keys():
            last_n_iter = state_dicts['n_iter']
        print(f"Loaded models from: {config.model_restore}!")

    # Start tensorboard logging
    writer = None
    if config.tb_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)

    # Start training
    training_loop(generator,
                  discriminator,
                  g_optimizer,
                  d_optimizer,
                  gan_loss_g,
                  gan_loss_d,
                  train_dataloader,
                  last_n_iter,
                  writer,
                  config)


if __name__ == '__main__':
    main()
