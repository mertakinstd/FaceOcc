import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from torchvision import transforms as TF
import tqdm

from faceocc_runtime import (
    build_faceocc_model,
    configure_cuda_performance,
    make_faceocc_input_preprocess,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path('./Dataset/FaceOcc/COFW_test')
TO_TENSOR = TF.ToTensor()

configure_cuda_performance()
input_preprocess = make_faceocc_input_preprocess(DEVICE)
model = build_faceocc_model(encoder_weights=None)
best_models = list(Path('./pretrained').glob('*_best.safetensors'))
if len(best_models) != 1:
    raise RuntimeError(f'Expected one best checkpoint in pretrained/, found {len(best_models)}')
model.load_state_dict(load_file(best_models[0], device='cpu'))
model.to(DEVICE)
model.eval()
images = os.listdir(ROOT / 'img')


def load_data(name):
    image = TO_TENSOR(Image.open(ROOT / 'img' / name)).unsqueeze(0)
    mask = cv2.imread(str(ROOT / 'mask' / f'{Path(name).stem}.png'), 0) // 255
    return image, mask


def calc_iou(pred, labeled):
    labeled = labeled.squeeze()
    pred = pred.squeeze()
    inter = (pred * labeled).sum()
    union = pred.sum() + labeled.sum() - inter
    return inter * 1.0 / union


def calc_acc(pred, labeled):
    labeled = labeled.squeeze()
    pred = pred.squeeze()
    return (pred == labeled).sum() / pred.size


def calc_recall(pred, labeled):
    labeled = labeled.squeeze()
    pred = pred.squeeze()
    return (pred * labeled).sum() / labeled.sum()


total_iou = 0
total_acc = 0
total_recall = 0
for name in tqdm.tqdm(images):
    data, gt_mask = load_data(name)
    with torch.no_grad():
        pred = model(input_preprocess(data.to(DEVICE)))
    pred_mask = (pred > 0).type(torch.int8).squeeze().cpu().numpy()
    total_iou += calc_iou(pred_mask, gt_mask)
    total_acc += calc_acc(pred_mask, gt_mask)
    total_recall += calc_recall(pred_mask, gt_mask)

print(f'iou={total_iou / len(images)}')
print(f'acc={total_acc / len(images)}')
print(f'recall={total_recall / len(images)}')

# FPS evaluation
dummy_input = input_preprocess(torch.rand(1, 3, 256, 256, device=DEVICE))
repetitions = 1000
with torch.no_grad():
    for _ in range(100):
        model(dummy_input)

if DEVICE.type == 'cuda':
    timings = np.zeros(repetitions)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in tqdm.tqdm(range(repetitions)):
            start.record()
            model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            timings[i] = start.elapsed_time(end)
    total_time = timings.sum() / 1000.0
else:
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(repetitions)):
            model(dummy_input)
    total_time = time.perf_counter() - start_time

print(f'fps={repetitions / total_time}')
