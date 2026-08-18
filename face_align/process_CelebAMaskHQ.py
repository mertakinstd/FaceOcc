#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from face_align.align import Preprocess

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw' / 'CelebAMask-HQ'
IMAGE_ROOT = RAW / 'CelebA-HQ-img'
MASK_ROOT = RAW / 'CelebAMask-HQ-mask-anno'
LANDMARK_ROOT = ROOT / 'data' / 'prepared' / 'landmarks'
OUTPUT_IMAGE_ROOT = ROOT / 'Dataset' / 'CelebA-HQ-align'
OUTPUT_MASK_ROOT = ROOT / 'Dataset' / 'CelebAMask-HQ-align'

ATTRIBUTES = [
    'skin', 'l_brow', 'r_brow', 'eye_g', 'l_eye', 'r_eye',
    'nose', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',
]


def get_mask(image_name):
    number = int(Path(image_name).stem)
    folder = str(number // 2000)
    mask = np.zeros((512, 512))
    for label, attribute in enumerate(ATTRIBUTES, 1):
        path = MASK_ROOT / folder / f'{number:05d}_{attribute}.png'
        if path.exists():
            part = np.array(Image.open(path).convert('P'))
            mask[part == 225] = label
    return mask


def process(image_name):
    image = cv2.imread(str(IMAGE_ROOT / image_name))
    image = cv2.resize(image, (512, 512))
    mask = get_mask(image_name)
    landmarks = np.load(LANDMARK_ROOT / f'{Path(image_name).stem}.npy', allow_pickle=False) * 0.5
    image, mask, _ = Preprocess(image, mask, landmarks)
    return image, mask


def main():
    for path in (IMAGE_ROOT, MASK_ROOT, LANDMARK_ROOT):
        if not path.is_dir():
            raise FileNotFoundError(path)

    OUTPUT_IMAGE_ROOT.mkdir(parents=True)
    OUTPUT_MASK_ROOT.mkdir(parents=True)

    landmarks = sorted(LANDMARK_ROOT.glob('*.npy'), key=lambda path: int(path.stem))
    if not landmarks:
        raise RuntimeError('No prepared landmarks found')

    for landmark_path in tqdm(landmarks, desc='Aligning CelebAMask-HQ', unit='image'):
        name = f'{landmark_path.stem}.jpg'
        image, mask = process(name)
        if not cv2.imwrite(str(OUTPUT_IMAGE_ROOT / name), image):
            raise OSError(f'Failed to write aligned image: {name}')
        mask_path = OUTPUT_MASK_ROOT / f'{Path(name).stem}.png'
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8)):
            raise OSError(f'Failed to write aligned mask: {mask_path.name}')


if __name__ == '__main__':
    main()
