import time
from collections import Counter

import cv2
import numpy as np
import os
import pathlib
import sys
import torch
from tqdm import tqdm

from Dataset.utils import tensor2img
from meter import AverageValueMeter


class Epoch:
    def __init__(
        self,
        model,
        loss,
        metrics,
        stage_name,
        device='cpu',
        sv_pth=None,
        show_step=500,
        verbose=True,
        diagnostics=None,
        collect_sampling=False,
        input_preprocess=None,
    ):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.it = 0
        self.show_step = show_step
        self.diagnostics = diagnostics
        self.collect_sampling = collect_sampling
        self.input_preprocess = input_preprocess
        if not sv_pth:
            pth = pathlib.Path(__file__).parent.absolute()
            sv_folder = 'res'
            sv_root = os.path.join(pth, sv_folder)
            self.sv_root = sv_root
            if not os.path.exists(sv_root):
                os.mkdir(sv_root)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    @staticmethod
    def _as_list(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _update_sampling(self, counter, metadata):
        if not metadata:
            return
        sample_kinds = self._as_list(metadata['sample_kind'])
        sources = self._as_list(metadata['source'])
        texture_replaced = self._as_list(metadata['texture_replaced'])
        mask_asset = self._as_list(metadata['mask_asset'])
        if not (len(sample_kinds) == len(sources) == len(texture_replaced) == len(mask_asset)):
            raise RuntimeError('sampling metadata batch fields have inconsistent lengths')

        for kind, source, texture_flag, mask_flag in zip(
            sample_kinds, sources, texture_replaced, mask_asset
        ):
            counter['samples_total'] += 1
            counter[f'sample_kind:{kind}'] += 1
            counter[f'source:{source}'] += 1
            if kind == 'synthetic':
                counter['synthetic_total'] += 1
                counter['synthetic_texture_replaced'] += int(bool(texture_flag))
                counter['synthetic_mask_asset'] += int(bool(mask_flag))

    @staticmethod
    def _sampling_logs(counter):
        total = counter['samples_total']
        if total == 0:
            return {}
        synthetic_total = counter['synthetic_total']
        safe_synthetic = max(synthetic_total, 1)
        return {
            'sampling_real_fraction': counter['sample_kind:real'] / total,
            'sampling_synthetic_fraction': counter['sample_kind:synthetic'] / total,
            'sampling_source_celeba_fraction': counter['source:celeba'] / total,
            'sampling_source_ffhq_fraction': counter['source:ffhq'] / total,
            'sampling_source_internet_fraction': counter['source:internet'] / total,
            'sampling_texture_replaced_given_synthetic': (
                counter['synthetic_texture_replaced'] / safe_synthetic
            ),
            'sampling_mask_asset_given_synthetic': counter['synthetic_mask_asset'] / safe_synthetic,
            'sampling_samples_total': float(total),
        }

    def run(self, dataloader):
        self.on_epoch_start()
        if self.diagnostics is not None:
            self.diagnostics.reset()

        logs = {}
        sampling_counter = Counter()
        loss_meter = AverageValueMeter()
        metric_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for batch in iterator:
                if len(batch) == 3:
                    x, y, metadata = batch
                elif len(batch) == 2:
                    x, y = batch
                    metadata = None
                else:
                    raise RuntimeError(f'unexpected dataloader batch with {len(batch)} fields')

                if self.collect_sampling:
                    self._update_sampling(sampling_counter, metadata)

                self.it += 1
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                model_x = self.input_preprocess(x) if self.input_preprocess is not None else x
                loss, y_pred = self.batch_update(model_x, y)
                loss_value = loss.item()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).item()
                    metric_meters[metric_fn.__name__].add(metric_value)

                metric_logs = {k: v.mean for k, v in metric_meters.items()}
                logs.update(metric_logs)

                if self.diagnostics is not None:
                    self.diagnostics.update(y_pred.detach(), y.detach())

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                if self.it % self.show_step == 0:
                    with torch.no_grad():
                        pred_mask = (y_pred > 0).type(torch.float32)
                        face_pred = tensor2img(x * pred_mask)
                        face_gt = tensor2img(x * y)
                        img = tensor2img(x)
                        show = np.concatenate((img, face_gt, face_pred), axis=0)
                        sv_name = f'{self.stage_name}_it_{self.it}.png'
                        sv_pth = os.path.join(self.sv_root, sv_name)
                        cv2.imwrite(sv_pth, show[:, :, ::-1])

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        logs['seconds'] = elapsed
        logs['it_s'] = len(dataloader) / elapsed if elapsed > 0 else float('nan')
        if self.device.type == 'cuda':
            logs['peak_vram_mib'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            logs['peak_vram_mib'] = 0.0

        if self.collect_sampling:
            logs.update(self._sampling_logs(sampling_counter))
        if self.diagnostics is not None:
            logs.update(self.diagnostics.compute())

        return logs


class TrainEpoch(Epoch):
    def __init__(
        self, model, loss, metrics, optimizer, device='cpu', verbose=True, input_preprocess=None
    ):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            collect_sampling=True,
            input_preprocess=input_preprocess,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(
        self, model, loss, metrics, device='cpu', verbose=True, diagnostics=None, input_preprocess=None
    ):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
            show_step=200,
            diagnostics=diagnostics,
            input_preprocess=input_preprocess,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model(x)
            loss = self.loss(prediction, y)
        return loss, prediction


if __name__ == '__main__':
    epoch = Epoch(None, None, None, None)
