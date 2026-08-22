import torch
import torch.nn.functional as F
import math
import cv2


class RandomAffine(object):
    def __init__(self, scale, angle, flip=0.5, translate=0.1):
        if not 0.0 <= translate <= 1.0:
            raise ValueError('translate must be a non-negative image fraction in [0, 1]')
        self.scale = scale
        self.angle = angle / 180 * math.pi
        self.flip = flip  # probability of left-right flip
        self.translate = translate

    def _sample_translation(self, height, width):
        """Sample independent symmetric x/y translations for affine_grid.

        ``translate`` follows torchvision-style semantics: it is the maximum
        displacement as a fraction of image width/height. ``affine_grid``
        expects translation in normalized coordinates, hence the conversion
        from sampled pixel displacement below.
        """
        if self.translate == 0.0:
            return 0.0, 0.0

        dx_pixels = (2.0 * torch.rand(1).item() - 1.0) * self.translate * width
        dy_pixels = (2.0 * torch.rand(1).item() - 1.0) * self.translate * height
        tx = 0.0 if width <= 1 else 2.0 * dx_pixels / (width - 1)
        ty = 0.0 if height <= 1 else 2.0 * dy_pixels / (height - 1)
        return tx, ty

    def __call__(self, data):
        # data = {'img': img, 'uv': uv, 'mat_inv': mat_inverse, 'mask': mask}
        img, mask = data['img'], data['mask']
        h, w = img.shape[1:]

        # flip flag
        flip = torch.rand(1).item() < self.flip

        # rotation matrix
        angle = 2 * self.angle * torch.rand(1) - self.angle
        cos = torch.cos(angle).item()
        sin = torch.sin(angle).item()

        s = (1 + 2 * self.scale * torch.rand(1) - self.scale).item()
        tx, ty = self._sample_translation(h, w)
        M = torch.tensor([
            [s * cos, s * sin, tx],
            [-s * sin, s * cos, ty]
        ], dtype=img.dtype)

        if flip:
            M[0, 0] *= -1
            M[1, 0] *= -1

        grid = F.affine_grid(M.unsqueeze(0), [1, 3, h, w], align_corners=True)
        image = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze(0)
        mask = F.grid_sample(mask.unsqueeze(0), grid, align_corners=True, mode='nearest').squeeze(0)
        image = torch.clamp(image, 0, 1)
        mask = (mask > 0).type(torch.float32)

        # image = cv2.warpAffine(img, M, (h, w))
        # mask = cv2.warpAffine(mask, M, (h, w), flags=cv2.INTER_NEAREST)

        data = {'img': image, 'mask': mask, }
        return data
