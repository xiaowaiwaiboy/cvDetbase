import numpy as np
from torch import optim
import collections
from CNN.data import *
from torchvision import transforms
import torch.utils.data
import argparse
from tensorboardX import SummaryWriter
import yaml
from CNN.model import build_detector
from CNN.utils import build_optimizer
import time
import os.path as osp
import os
from utils import json_file
import torch
from utils import collect_env, get_root_logger
from eval_model import mAP_compute

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='config/fcos_fpn_r50.yaml')
parser.add_argument('--resume-from', type=str, default=None)
parser.add_argument('--epochs', type=int, default=2, help='number of total epochs')
parser.add_argument('--save_epochs', type=int, default=1, help='when to save weights')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_gpu', type=str, default='0')
parser.add_argument('--log_iter', type=int, default=20)
parser.add_argument('--eval_epoch', type=int, default=1)
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
if opt.resume_from is not None:
    config = json_file.load(opt.resume_from)
    weights = torch.load(f'epoch_{config["epoch"]}')
else:
    config = yaml.load(open(opt.config_file, 'r'), Loader=yaml.FullLoader)
    weights = None

# json_file.show(config)
model_cfg = config.get('detector')
train_cfg = config.get('dataset_train')
eval_cfg = config.get('dataset_eval')

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logpath = f'runs/{model_cfg["cfg"]}_{train_cfg["cfg"]}/{timestamp}'.lower()
if not osp.exists(logpath):
    os.makedirs(logpath)
json_file.save(config, osp.join(logpath, f'{timestamp}.json'))
writer = SummaryWriter(logdir=logpath, filename_suffix=f'_{timestamp}')
log_file = osp.join(logpath, f'{timestamp}.log')
logger = get_root_logger(name=f'{model_cfg["cfg"]}_{train_cfg["cfg"]}'.lower(), log_file=log_file, log_level='INFO',
                         format_='%(message)s')
env_info_dict = collect_env()
env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

model = build_detector(cfg=model_cfg)
if weights:
    model.load_state_dict(weights)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
else:
    model = torch.nn.DataParallel(model)


train_dataset = build_dataset(cfg=train_cfg, transform=transforms.Compose([Augmenter(),
                                                                          Resizer(img_sizes=train_cfg.get('img_sizes'))]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.get('batch_size'), collate_fn=collate_fn)

eval_dataset = build_dataset(cfg=eval_cfg, transform=Resizer(img_sizes=eval_cfg.get('img_sizes')))
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_cfg.get('batch_size'), collate_fn=collate_fn)

optimizer = build_optimizer(config.get('optimizer'), params=model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_hist = collections.deque(maxlen=500)
best_map = 0
for epoch in range(opt.epochs):
    model.training = True
    model.train()

    epoch_loss = []

    for batch_step, data in enumerate(train_loader):
        global_steps = len(train_loader) * epoch + batch_step
        images, boxes, classes = data['img'], data['annot'][..., :4], data['annot'][..., -1]
        optimizer.zero_grad()
        for pram in optimizer.param_groups:
            lr = pram['lr']

        losses = model([images.cuda(), boxes.cuda(), classes.cuda()])
        cls_loss = losses[0].mean()
        reg_loss = losses[1].mean()
        total_loss = losses[-1].mean()
        if bool(total_loss == 0):
            continue

        writer.add_scalar('lr', lr, global_step=global_steps)
        writer.add_scalar('cls_loss', cls_loss.mean(), global_step=global_steps)
        writer.add_scalar('reg_loss', reg_loss.mean(), global_step=global_steps)
        writer.add_scalar('total_loss', total_loss.mean(), global_step=global_steps)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        loss_hist.append(float(total_loss))
        epoch_loss.append(float(total_loss))

        if batch_step % opt.log_iter == 0:
            logger.info(
                'Epoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss:'
                '{:1.5f}'.format(epoch+1, batch_step, len(train_loader), float(cls_loss), float(reg_loss), np.mean(loss_hist))
            )
        del losses

    if epoch % opt.eval_epoch == 0:
        model.training = False
        model.eval()
        mAP = mAP_compute(model, eval_loader, logger, iou_thresh=0.7)
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), '{}/epoch_{}_{:.3f}.pth'.format(logpath, epoch+1, mAP))
            config['epoch'] = epoch
            json_file.save(config, osp.join(logpath, f'{timestamp}.json'))

    scheduler.step(np.mean(epoch_loss))
writer.export_scalars_to_json(osp.join(logpath, f'{timestamp}_scalars.json'))
model.eval()
