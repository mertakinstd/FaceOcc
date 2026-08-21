from __future__ import annotations

import os
import random

import numpy as np
import segmentation_models_pytorch as smp
import torch

ENCODER = 'resnet18'
ENCODER_DEPTH = 5
ENCODER_WEIGHTS = 'imagenet'
DECODER_CHANNELS = (256, 128, 64, 32, 16)
DECODER_NORM = 'batchnorm'
DECODER_INTERPOLATION = 'nearest'
DEFAULT_SEED = 42


def seed_everything(seed: int) -> None:
    """Seed host and torch RNGs for a reproducible stochastic training stream."""

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy inside each DataLoader worker from PyTorch's worker seed."""

    del worker_id  # the worker-specific seed is already encoded by torch.initial_seed()
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def configure_cuda_performance() -> None:
    """Keep the modernized TF32 policy while making the run deterministic."""

    if not torch.cuda.is_available():
        return

    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def cuda_protocol_metadata() -> dict:
    metadata = {
        'precision_mode': 'tf32',
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                'fp32_precision': str(torch.backends.fp32_precision),
                'matmul_fp32_precision': str(torch.backends.cuda.matmul.fp32_precision),
                'cudnn_fp32_precision': str(torch.backends.cudnn.fp32_precision),
                'conv_fp32_precision': str(torch.backends.cudnn.conv.fp32_precision),
            }
        )
    return metadata


def build_faceocc_model(*, encoder_weights=ENCODER_WEIGHTS):
    return smp.Unet(
        encoder_name=ENCODER,
        encoder_depth=ENCODER_DEPTH,
        encoder_weights=encoder_weights,
        decoder_use_norm=DECODER_NORM,
        decoder_channels=DECODER_CHANNELS,
        decoder_attention_type=None,
        decoder_interpolation=DECODER_INTERPOLATION,
        in_channels=3,
        classes=1,
        activation=None,
    )
