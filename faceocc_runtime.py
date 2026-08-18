"""Shared FaceOcc model and CUDA runtime configuration."""

import segmentation_models_pytorch as smp
import torch

ENCODER = "resnet18"
ENCODER_DEPTH = 5
ENCODER_WEIGHTS = "imagenet"
DECODER_CHANNELS = (256, 128, 64, 32, 16)
DECODER_NORM = "batchnorm"
DECODER_INTERPOLATION = "nearest"
ATTENTION = None
IN_CHANNELS = 3
CLASSES = 1
ACTIVATION = None


def configure_cuda_performance() -> None:
    """Use the agreed FP32/TF32 performance profile on CUDA.

    Model parameters and optimizer state remain FP32. TF32 is allowed only as
    the internal compute precision for CUDA matmul and cuDNN convolution.
    cuDNN benchmarking is enabled because FaceOcc trains at a fixed 256x256
    resolution. Deterministic mode remains disabled; explicit seeding is a
    separate future reproducibility feature.
    """
    if not torch.cuda.is_available():
        return

    # PyTorch >= 2.9 fine-grained TF32 controls. Keep the generic FP32 policy
    # IEEE and opt the expensive CUDA operators into TF32 explicitly.
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def build_faceocc_model(*, encoder_weights=ENCODER_WEIGHTS):
    """Build the modern SMP 0.5 FaceOcc U-Net with explicit architecture knobs."""
    return smp.Unet(
        encoder_name=ENCODER,
        encoder_depth=ENCODER_DEPTH,
        encoder_weights=encoder_weights,
        decoder_use_norm=DECODER_NORM,
        decoder_channels=DECODER_CHANNELS,
        decoder_attention_type=ATTENTION,
        decoder_interpolation=DECODER_INTERPOLATION,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
