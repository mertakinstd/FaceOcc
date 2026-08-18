import argparse
import cv2
import os
import pathlib
import sys
import numpy as np
import pickle
from PIL import Image
import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from face_align.align import Preprocess
from face_align.visualization import vis_parsing_maps
RAW_ROOT = REPO_ROOT / 'data' / 'raw' / 'CelebAMask-HQ'

# Defaults match setup.sh's deterministic raw-data layout. They remain module
# globals because the original helper functions read them directly.
root_img = str(RAW_ROOT / 'CelebA-HQ-img')
root_mask = str(RAW_ROOT / 'CelebAMask-HQ-mask-anno')
root_ldmk = str(RAW_ROOT / 'ldmk_init')  # landmarks detected by 3DDFA_V2
root_img_sv = str(REPO_ROOT / 'Dataset' / 'CelebA-HQ-align')
root_mask_sv = str(REPO_ROOT / 'Dataset' / 'CelebAMask-HQ-align')

atts = ['skin', 'l_brow', 'r_brow', 'eye_g', 'l_eye', 'r_eye', 'nose', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat']


def get_mask(img_name):
    num = int(img_name.split('.')[0])
    folder = str(num // 2000)
    mask = np.zeros((512, 512))
    for l, att in enumerate(atts, 1):
        file_name = ''.join([str(num).rjust(5, '0'), '_', att, '.png'])
        path = os.path.join(root_mask, folder, file_name)
        if os.path.exists(path):
            sep_mask = np.array(Image.open(path).convert('P'))
            mask[sep_mask == 225] = l
    return mask


def get_img(name):
    pth = os.path.join(root_img, name)
    I = cv2.imread(pth)
    I = cv2.resize(I, (512, 512))  # resize the image from 1024x1024 to 512x512
    return I


def get_ldmk(name):
    ldmk_name = name.split('.')[0] + '.pkl'
    pth = os.path.join(root_ldmk, ldmk_name)
    with open(pth, 'rb') as f:
        ldmk = pickle.load(f)
    ldmk = ldmk * 0.5  # original image is 1024x1024 while seg map is 512x512
    return ldmk


def process(img_name):
    img = get_img(img_name)
    seg = get_mask(img_name)
    ldmk = get_ldmk(img_name)
    img, seg, ldmk = Preprocess(img, seg, ldmk)
    return img, seg, ldmk


def parse_args():
    parser = argparse.ArgumentParser(description='Align CelebAMask-HQ for FaceOcc training.')
    parser.add_argument('--image-root', default=root_img)
    parser.add_argument('--mask-root', default=root_mask)
    parser.add_argument('--landmark-root', default=root_ldmk)
    parser.add_argument('--output-image-root', default=root_img_sv)
    parser.add_argument('--output-mask-root', default=root_mask_sv)
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show the legacy OpenCV visualization (requires a GUI-enabled OpenCV build and display).',
    )
    return parser.parse_args()


def main():
    global root_img, root_mask, root_ldmk, root_img_sv, root_mask_sv

    args = parse_args()
    root_img = os.path.abspath(args.image_root)
    root_mask = os.path.abspath(args.mask_root)
    root_ldmk = os.path.abspath(args.landmark_root)
    root_img_sv = os.path.abspath(args.output_image_root)
    root_mask_sv = os.path.abspath(args.output_mask_root)

    for required, label in (
        (root_img, 'CelebA-HQ image root'),
        (root_mask, 'CelebAMask-HQ mask root'),
        (root_ldmk, '3DDFA landmark root'),
    ):
        if not os.path.isdir(required):
            raise FileNotFoundError(f'{label} not found: {required}')

    os.makedirs(root_img_sv, exist_ok=True)
    os.makedirs(root_mask_sv, exist_ok=True)

    for name in tqdm.tqdm(os.listdir(root_img)):
        img, seg, ldmk = process(name)

        if args.preview:
            img_mask = vis_parsing_maps(img, seg, 1)
            show = np.concatenate((img, img_mask))
            cv2.imshow('seg', show)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

        img_sv_pth = os.path.join(root_img_sv, name)
        cv2.imwrite(img_sv_pth, img)

        seg_name = name.split('.')[0] + '.pkl'
        seg_sv_pth = os.path.join(root_mask_sv, seg_name)
        with open(seg_sv_pth, 'wb') as f:
            pickle.dump(seg, f)

    if args.preview:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
