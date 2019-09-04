""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
    Let's go.
"""
import datetime
import time

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

import dataset
import BigGAN
import train_fns
import utils
from utils import activation_dict, prepare_z_y, save_and_sample, prepare_parser


def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = 64
    config['n_classes'] = 120
    config['G_activation'] = activation_dict[config['G_nl']]
    config['D_activation'] = activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'
    # Seed RNG
    utils.seed_rng(config['seed'])
    # Prepare root folders if necessary
    utils.prepare_root(config)
    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    G = BigGAN.Generator(**config).to(device)
    D = BigGAN.Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = BigGAN.Generator(**{**config, 'skip_init': True,
                                    'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    GD = BigGAN.G_D(G, D)
    print(G)
    print(D)
    print('Number of params in G: {} D: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'config': config}

    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
    loaders = dataset.get_data_loaders(
        data_root=INPUT_PATH,
        batch_size=D_batch_size,
        num_workers=config['num_workers'],
        shuffle=config['shuffle'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = prepare_z_y(
        G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = prepare_z_y(
        G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    # Loaders are loaded, prepare the training function
    train = train_fns.create_train_fn(G, D, GD, z_, y_, ema, state_dict, config)

    print('Beginning training at epoch %d...' % state_dict['epoch'])
    start_time = time.perf_counter()
    total_iters = config['num_epochs'] * len(loaders[0])

    # Train for specified number of epochs, although we mostly track G iterations.
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        for i, (x, y) in enumerate(loaders[0]):
            # Increment the iteration counter
            state_dict['itr'] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            metrics = train(x, y)

            if not (state_dict['itr'] % config['log_interval']):
                curr_time = time.perf_counter()
                curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%H:%M:%S')
                elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
                log = (
                        "[{}] [{}] [{} / {}] Ep {}, ".format(curr_time_str, elapsed, state_dict['itr'], total_iters,
                                                             epoch) +
                        ', '.join(['%s : %+4.3f' % (key, metrics[key]) for key in metrics])
                )
                print(log)

            # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    # if config['ema']:
                    # G_ema.eval()
                save_and_sample(G, G_ema, fixed_z, fixed_y, config)

            if config['stop_after'] > 0 and int(time.perf_counter() - start_time) > config['stop_after']:
                print("Time limit reached! Stopping training!")
                return

        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1


def main():
    # parse command line and run
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
