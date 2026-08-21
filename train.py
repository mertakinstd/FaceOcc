from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader

from Dataset.dataset import COFW_test, FaceMask
from diagnostics import ValidationDiagnostics
from faceocc_runtime import (
    DEFAULT_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_NORMALIZATION,
    build_faceocc_model,
    configure_cuda_performance,
    cuda_protocol_metadata,
    make_dataloader_generator,
    make_faceocc_input_preprocess,
    seed_everything,
    seed_worker,
)
from loss import DiceLoss, IoU, OhemBCELoss, Precision
from train_utils import TrainEpoch, ValidEpoch


BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 30
LEGACY_IOU_THRESHOLD_LOGIT = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description='Train the modernized FaceOcc baseline.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--allow-existing-run-dir', action='store_true')
    return parser.parse_args()


def make_run_dir(args) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = Path('runs') / f'resnet18_unet_ohembce_fp32_imagenetnorm_seed{args.seed}'
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_existing_run_dir:
        raise RuntimeError(
            f'Run directory is not empty: {run_dir}. '
            'Use a new --run-dir or pass --allow-existing-run-dir intentionally.'
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def flatten_epoch_row(epoch, optimizer, train_logs, valid_logs):
    row = {
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
    }
    row.update({f'train_{key}': value for key, value in train_logs.items()})
    row.update({f'val_{key}': value for key, value in valid_logs.items()})
    return row


def save_state_dict(model, path: Path) -> None:
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model_to_save.state_dict().items()
    }
    save_file(state_dict, str(path), metadata={'format': 'pt'})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for FaceOcc training; CPU training is intentionally unsupported.')

    # PYTHONHASHSEED must ideally be exported before interpreter startup; train.sh
    # does that. seed_everything still records and seeds all runtime RNGs here.
    seed_everything(args.seed)
    configure_cuda_performance()

    run_dir = make_run_dir(args)
    checkpoint_dir = run_dir / 'checkpoints'
    history_path = run_dir / 'history.csv'

    train_dataset = FaceMask()
    valid_dataset = COFW_test()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(args.seed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=True,  # legacy behavior; deterministic via the explicit generator
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(args.seed + 1),
    )

    device = torch.device('cuda')
    input_preprocess = make_faceocc_input_preprocess(device)
    model = build_faceocc_model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} CUDA devices via DataParallel')
    else:
        print(f'Using CUDA device 0: {torch.cuda.get_device_name(0)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    loss_bce = OhemBCELoss(thresh=0.7, n_min=256 ** 2 - 1).to(device)
    loss_dice = DiceLoss().to(device)  # retained but intentionally inactive, matching FaceOcc

    def criterion(pred, gt):
        bce = loss_bce(pred, gt)
        # dice = loss_dice(pred, gt)
        return bce
        # return dice

    metrics = [IoU(threshold=LEGACY_IOU_THRESHOLD_LOGIT), Precision(threshold=0.0)]
    diagnostics = ValidationDiagnostics(boundary_dilation_ratio=0.02, ece_bins=15)

    train_epoch = TrainEpoch(
        model=model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
        input_preprocess=input_preprocess,
    )
    valid_epoch = ValidEpoch(
        model=model,
        loss=criterion,
        metrics=metrics,
        device=device,
        verbose=True,
        diagnostics=diagnostics,
        input_preprocess=input_preprocess,
    )

    config = {
        'seed': args.seed,
        'precision': 'fp32',
        'normalization': INPUT_NORMALIZATION,
        'input_range_before_normalization': '[0,1]',
        'normalization_stage': 'after augmentation, immediately before model forward',
        'imagenet_mean': list(IMAGENET_MEAN),
        'imagenet_std': list(IMAGENET_STD),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'persistent_workers': False,
        'train_size': len(train_dataset),
        'validation_size': len(valid_dataset),
        'primary_metric': 'val_iou_score',
        'primary_metric_logit_threshold': LEGACY_IOU_THRESHOLD_LOGIT,
        'checkpoint_selection': 'max legacy validation IoU',
        'loss': 'legacy OHEM-BCE',
        'ohem_thresh': 0.7,
        'ohem_n_min': 256 ** 2 - 1,
        'diagnostics': {
            'boundary_dilation_ratio': diagnostics.boundary_dilation_ratio,
            'ece_bins': diagnostics.ece_bins,
            'probability_thresholds': list(diagnostics.probability_thresholds),
        },
        'cuda_protocol': cuda_protocol_metadata(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(0),
        'cuda_device_count': torch.cuda.device_count(),
        'python_version': platform.python_version(),
    }
    write_json(run_dir / 'run_config.json', config)

    max_score = float('-inf')
    best_epoch = None
    best_row = None
    history_writer = None
    history_handle = history_path.open('w', newline='')
    total_started = time.perf_counter()

    try:
        for epoch in range(1, EPOCHS + 1):
            print(
                f'\n Epoch: {epoch}/{EPOCHS} '
                f'[fp32, normalization={INPUT_NORMALIZATION}, seed={args.seed}]'
            )
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            row = flatten_epoch_row(epoch, optimizer, train_logs, valid_logs)

            if history_writer is None:
                history_writer = csv.DictWriter(history_handle, fieldnames=list(row.keys()))
                history_writer.writeheader()
            history_writer.writerow(row)
            history_handle.flush()

            current_score = valid_logs['iou_score']
            if max_score < current_score:
                print('best model')
                max_score = current_score
                best_epoch = epoch
                best_row = dict(row)
                model_name = f'epoch_{epoch}_best.safetensors'
                for path in checkpoint_dir.glob('*_best.safetensors'):
                    path.rename(path.with_name('_'.join(path.name.split('_')[:2]) + '.safetensors'))
            else:
                model_name = f'epoch_{epoch}.safetensors'

            model_path = checkpoint_dir / model_name
            save_state_dict(model, model_path)
            print(f'Epoch: {epoch}, model saved to {model_path}')
            for path in checkpoint_dir.glob('*.safetensors'):
                if 'best' not in path.name and path.name != model_name:
                    path.unlink()

            if epoch == 20:  # exact legacy FaceOcc schedule
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease learning rate to 1e-5')
    finally:
        history_handle.close()

    total_seconds = time.perf_counter() - total_started
    summary = {
        'seed': args.seed,
        'precision': 'fp32',
        'normalization': INPUT_NORMALIZATION,
        'best_epoch': best_epoch,
        'best_val_iou': max_score,
        'total_seconds': total_seconds,
        'best_epoch_metrics': best_row,
        'history_csv': str(history_path),
        'checkpoint_dir': str(checkpoint_dir),
    }
    write_json(run_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
