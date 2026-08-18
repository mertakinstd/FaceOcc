import torch
from torch import nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from Dataset.dataset import FaceMask, COFW_test
from loss import OhemBCELoss, DiceLoss, IoU, Precision
from train_utils import TrainEpoch, ValidEpoch
from faceocc_runtime import build_faceocc_model, configure_cuda_performance
import os

train_dataset = FaceMask()
valid_dataset = COFW_test()
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
)

epochs = 30
model_root = './pretrained/'
if not os.path.exists(model_root):
    os.mkdir(model_root)

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for FaceOcc training; CPU training is intentionally unsupported.')

DEVICE = torch.device('cuda')
configure_cuda_performance()

model = build_faceocc_model()
model = model.to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f'Using {torch.cuda.device_count()} CUDA devices via DataParallel')
else:
    print(f'Using CUDA device 0: {torch.cuda.get_device_name(0)}')
# state_dict = torch.load('pretrained/epoch_26_best.ckpt')
# model.load_state_dict(state_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)

loss_bce = OhemBCELoss(thresh=0.7, n_min=256 ** 2-1).to(DEVICE)
loss_dice = DiceLoss().to(DEVICE)


def criterion(pred, gt):
    bce = loss_bce(pred, gt)
    # dice = loss_dice(pred, gt)
    return bce
    # return dice


metrics = [IoU(threshold=0.0), Precision(threshold=0.0)]

train_epoch = TrainEpoch(
    model=model,
    loss=criterion,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True
)

valid_epoch = ValidEpoch(
    model=model,
    loss=criterion,
    metrics=metrics,
    device=DEVICE,
    verbose=True
)

max_score = 0
for epoch in range(1, epochs+1):
    print(f'\n Epoch: {epoch}/{epochs}')
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if max_score < valid_logs['iou_score']:
        print('best model')
        max_score = valid_logs['iou_score']
        model_name = f'epoch_{epoch}_best.safetensors'
        for name in os.listdir(model_root):
            if name.endswith('.safetensors') and 'best' in name:
                to_rename = os.path.join(model_root, name)
                new_name = '_'.join(name.split('_')[:2])+'.safetensors'
                new_name = os.path.join(model_root, new_name)
                os.rename(to_rename, new_name)

    else:
        model_name = f'epoch_{epoch}.safetensors'

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model_to_save.state_dict().items()
    }
    model_pth = os.path.join(model_root, model_name)
    save_file(state_dict, model_pth, metadata={'format': 'pt'})
    print(f'Epoch: {epoch}, model saved')
    for name in os.listdir(model_root):
        if name.endswith('.safetensors') and ('best' not in name) and (name != model_name):
            to_remove = os.path.join(model_root, name)
            os.remove(to_remove)

    if epoch == 20: # decrease the learning rate to 1e-5 from epoch 20
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5')
