import torch
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from depth2mesh import config, data
from depth2mesh.checkpoints import CheckpointIO
from depth2mesh.utils.logs import create_logger
from depth2mesh.utils.sampler import GroupedFixedSampler, GroupedRandomSampler, GroupedBatchSampler

from collections import OrderedDict

# Arguments
parser = argparse.ArgumentParser(
    description='Training function.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--validate-every-epoch', action='store_true', help='Whether to validate every epoch or not.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for train and val loaders.')
parser.add_argument('--epochs-per-run', type=int, default=-1,
                    help='Number of epochs to train before restart.')

if  __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    num_workers = args.num_workers
    epochs_per_run = args.epochs_per_run

    # Set t0
    t0 = time.time()

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']
    inner_batch_size = cfg['training']['inner_batch_size']
    backup_every = cfg['training']['backup_every']
    exit_after = args.exit_after

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize model
    model = config.get_model(cfg, device='cuda').to(device)
    if cfg['training']['stage'] != 'skinning_weights':
        # Load skinning networks
        optim_skinning_net_path = cfg['model']['skinning_net1']
        ckpt = torch.load(optim_skinning_net_path)

        encoder_fwd_state_dict = OrderedDict()
        skinning_decoder_fwd_state_dict = OrderedDict()
        encoder_bwd_state_dict = OrderedDict()
        skinning_decoder_bwd_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            # Remove module key words, which were created by torch.nn.DataParallel
            if k.startswith('module'):
                k = k[7:]

            # Create state dicts for different modules
            if k.startswith('skinning_decoder_fwd'):
                skinning_decoder_fwd_state_dict[k[21:]] = v
            elif k.startswith('skinning_decoder_bwd'):
                skinning_decoder_bwd_state_dict[k[21:]] = v
            elif k.startswith('encoder_fwd'):
                encoder_fwd_state_dict[k[12:]] = v
            elif k.startswith('encoder_bwd'):
                encoder_bwd_state_dict[k[12:]] = v

        # Load state dicts
        model.encoder_fwd.load_state_dict(encoder_fwd_state_dict)
        model.encoder_bwd.load_state_dict(encoder_bwd_state_dict)
        model.skinning_decoder_fwd.load_state_dict(skinning_decoder_fwd_state_dict)
        model.skinning_decoder_bwd.load_state_dict(skinning_decoder_bwd_state_dict)

    # Intialize optimizer
    lr = cfg['training']['lr']
    if cfg['training']['stage'] == 'meta-hyper' and cfg['model']['decoder'] == 'hyper_bvp':
        if model.decoder.hierarchical_pose:
            optimizer = optim.Adam(
                params = [
                    {
                        "params": model.decoder.net.parameters(),
                        "lr": lr,
                    },
                    {
                        "params": model.decoder.pose_encoder.parameters(),
                        "lr": 1e-4,
                    }
                ]
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create trainer
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    # Dataset
    train_dataset = config.get_dataset('train', cfg)
    val_dataset = config.get_dataset('val', cfg, subsampling_rate=10) # use a fraction of data for quick validation

    if cfg['training']['stage'] in ['skinning_weights', 'meta']:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)
    elif cfg['training']['stage'] in ['meta-hyper']:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=num_workers,
            batch_sampler=GroupedBatchSampler(GroupedRandomSampler(train_dataset.indices, max_batch_size=batch_size)),
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)
    else:
        raise ValueError('Unsupported stage option. Supported stages are: skinning_weights, meta, meta-hyper')

    if cfg['training']['stage'] in ['skinning_weights', 'meta']:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=1, shuffle=False,
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)
    elif cfg['training']['stage'] in ['meta-hyper']:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, num_workers=num_workers,
            batch_sampler=GroupedBatchSampler(GroupedFixedSampler(val_dataset.indices, batch_size=inner_batch_size)),
            collate_fn=data.collate_remove_none,
            worker_init_fn=data.worker_init_fn)
    else:
        raise ValueError('Unsupported stage option. Supported stages are: skinning_weights, meta, meta-hyper')

    logger, writter = create_logger(out_dir)

    logger.info('Current best validation metric (%s): %.8f'
                % (model_selection_metric, metric_val_best))

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']
    max_iterations = cfg['training']['max_iterations']
    max_epochs = cfg['training']['max_epochs']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    logger.info('Total number of parameters: %d' % nparameters)
    logger.info (len(train_loader))

    curr_epoch_cnt = 0

    while epochs_per_run <= 0 or curr_epoch_cnt < epochs_per_run:
        epoch_it += 1
        for batch in train_loader:
            it += 1
            loss_dict = trainer.train_step(batch, it)
            loss = loss_dict['total_loss']
            for k, v in loss_dict.items():
                writter.add_scalar('train/{}'.format(k), v, it)

            # Print output
            if print_every > 0 and (it % print_every) == 0:
                logger.info('[Epoch %02d] it=%03d, loss=%.4f'
                      % (epoch_it, it, loss))

            # Save checkpoint
            if (checkpoint_every > 0 and it > 0 and (it % checkpoint_every) == 0):
                logger.info('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Backup if necessary
            if (backup_every > 0 and it > 0 and (it % backup_every) == 0):
                logger.info('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader, val_dataset)
                metric_val = eval_dict[model_selection_metric]
                logger.info('Validation metric (%s): %.4f'
                      % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    writter.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger.info('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)

            # Exit if necessary
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                logger.info('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

                exit(3)

            if max_iterations > 0 and it >= max_iterations:
                logger.info('Maximum iteration reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

                exit(0)

            if max_epochs > 0 and epoch_it >= max_epochs:
                logger.info('Maximum epoch reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

                exit(0)

        curr_epoch_cnt += 1

        # Run validation
        if args.validate_every_epoch:
            eval_dict = trainer.evaluate(val_loader, val_dataset)
            metric_val = eval_dict[model_selection_metric]
            logger.info('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                writter.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                logger.info('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

    logger.info('Job will restart soon. Saving checkpoint.')
    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                       loss_val_best=metric_val_best)

    exit(3)
