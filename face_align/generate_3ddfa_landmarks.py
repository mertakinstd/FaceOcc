#!/usr/bin/env python3
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
import onnxruntime as ort
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
THREEDDFA = ROOT / '.tools' / '3DDFA_V2'
IMAGE_ROOT = ROOT / 'data' / 'raw' / 'CelebAMask-HQ' / 'CelebA-HQ-img'
OUTPUT_ROOT = ROOT / 'data' / 'prepared' / 'landmarks'
CUDA_PROVIDER = 'CUDAExecutionProvider'


def cpu_nms(dets, threshold):
    if len(dets) == 0:
        return []

    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = w * h
        iou = overlap / (areas[i] + areas[rest] - overlap)
        order = rest[iou < threshold]

    return keep


def install_3ddfa_compatibility():
    if 'long' not in np.__dict__:
        np.long = np.int64

    shim = types.ModuleType('FaceBoxes.utils.nms.cpu_nms')
    shim.cpu_nms = cpu_nms
    def cpu_soft_nms(*_args, **_kwargs):
        raise NotImplementedError

    shim.cpu_soft_nms = cpu_soft_nms
    sys.modules[shim.__name__] = shim
    sys.path.insert(0, str(THREEDDFA))


install_3ddfa_compatibility()

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX  # noqa: E402
from bfm.bfm import BFMModel  # noqa: E402
from utils.functions import crop_img, parse_roi_box_from_bbox  # noqa: E402
from utils.io import _load  # noqa: E402
from utils.tddfa_util import _parse_param, similar_transform  # noqa: E402


def cuda_session(path):
    if CUDA_PROVIDER not in ort.get_available_providers():
        raise RuntimeError('ONNX Runtime CUDAExecutionProvider is unavailable')
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=[CUDA_PROVIDER, 'CPUExecutionProvider']
    )
    if session.get_providers()[0] != CUDA_PROVIDER:
        raise RuntimeError('Failed to initialize ONNX Runtime CUDAExecutionProvider')
    return session


class CUDAFaceBoxes(FaceBoxes_ONNX):
    def __init__(self):
        self.session = cuda_session(THREEDDFA / 'FaceBoxes' / 'weights' / 'FaceBoxesProd.onnx')
        self.timer_flag = False


class SparseTDDFA:
    def __init__(self, config):
        self.size = int(config.get('size', 120))
        bfm = BFMModel(
            str(THREEDDFA / config.get('bfm_fp', 'configs/bfm_noneck_v3.pkl')),
            shape_dim=int(config.get('shape_dim', 40)),
            exp_dim=int(config.get('exp_dim', 10)),
        )
        self.u_base = bfm.u_base
        self.w_shp_base = bfm.w_shp_base
        self.w_exp_base = bfm.w_exp_base

        params = _load(str(THREEDDFA / f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl'))
        self.param_mean = params['mean']
        self.param_std = params['std']
        self.session = cuda_session(THREEDDFA / 'weights' / 'mb1_120x120.onnx')

    def landmarks(self, image, box):
        roi_box = parse_roi_box_from_bbox(box)
        crop = crop_img(image, roi_box)
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = crop.astype(np.float32).transpose(2, 0, 1)[None]
        tensor = (tensor - 127.5) / 128.0
        param = self.session.run(None, {'input': tensor})[0].ravel().astype(np.float32)
        param = param * self.param_std + self.param_mean

        rotation, offset, shape, expression = _parse_param(param)
        points = rotation @ (
            self.u_base + self.w_shp_base @ shape + self.w_exp_base @ expression
        ).reshape(3, -1, order='F') + offset
        points = similar_transform(points, roi_box, self.size)
        return points[:2].T.astype(np.float32, copy=False)


def main():
    if not THREEDDFA.is_dir():
        raise FileNotFoundError(f'3DDFA_V2 not found: {THREEDDFA}')
    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f'CelebA-HQ images not found: {IMAGE_ROOT}')

    images = sorted(IMAGE_ROOT.glob('*.jpg'))
    if len(images) != 30000:
        raise RuntimeError(f'Expected 30,000 CelebA-HQ images, found {len(images)}')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for landmark preparation')

    with (THREEDDFA / 'configs' / 'mb1_120x120.yml').open(encoding='utf-8') as handle:
        model = SparseTDDFA(yaml.safe_load(handle))
    detector = CUDAFaceBoxes()
    OUTPUT_ROOT.mkdir(parents=True)

    detected = 0
    missed = 0
    for image_path in tqdm(images, desc='3DDFA landmarks', unit='image'):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to read {image_path}')

        boxes = detector(image)
        if len(boxes) == 0:
            missed += 1
            continue

        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        landmarks = model.landmarks(image, box)
        if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
            raise RuntimeError(f'Invalid landmarks for {image_path}')
        np.save(OUTPUT_ROOT / f'{image_path.stem}.npy', landmarks, allow_pickle=False)
        detected += 1

    total = len(images)
    print(f'3DDFA landmarks: {detected}/{total} successful ({detected / total:.2%}), {missed} undetected')


if __name__ == '__main__':
    main()
