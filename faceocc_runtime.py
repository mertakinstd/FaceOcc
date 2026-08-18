import segmentation_models_pytorch as smp
import torch

ENCODER = 'resnet18'
ENCODER_DEPTH = 5
ENCODER_WEIGHTS = 'imagenet'
DECODER_CHANNELS = (256, 128, 64, 32, 16)
DECODER_NORM = 'batchnorm'
DECODER_INTERPOLATION = 'nearest'


def configure_cuda_performance():
    if not torch.cuda.is_available():
        return
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


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
