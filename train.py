from CNN.data import *
from torchvision import transforms
import torch.utils.data
import argparse
from tensorboardX import SummaryWriter
import yaml
from CNN.model import build_detector
from CNN.utils import build_optimizer
import time
import os
from utils import json_file
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='config/efficientdet_fpn.yaml')
parser.add_argument('--resume-from', type=str, default=None)
parser.add_argument('--epochs', type=int, default=20, help='number of total epochs')
parser.add_argument('--save_epochs', type=int, default=10, help='when to save weights')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--warmup_steps', type=int, default=500)
parser.add_argument('--optimizer', type=str, default='SGD')
opt = parser.parse_args()

if opt.resume_from is not None:
    config = json_file.load(opt.resume_from + 'cfg.json')
    weights = torch.load(f'epoch_{config["epoch"]}')
else:
    config = yaml.load(open(opt.config_file, 'r'), Loader=yaml.FullLoader)
    weights = None

# json_file.show(config)
model_cfg = config.get('detector')
data_cfg = config.get('dataset_train')

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logpath = f'runs/{timestamp}_{model_cfg["cfg"]}_{data_cfg["cfg"]}'.lower()
writer = SummaryWriter(logdir=logpath)
json_file.save(config, logpath)

model = build_detector(cfg=model_cfg)
if weights:
    model.load_state_dict(weights)
model = torch.nn.DataParallel(model)
model = model.cuda().train()

train_dataset = build_dataset(cfg=data_cfg,
                              transform=transforms.Compose([Augmenter(), Resizer(img_sizes=data_cfg.get('img_sizes'))]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_cfg.get('batch_size'), collate_fn=collate_fn)

optimizer = build_optimizer(config.get('optimizer'), params=model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)

for epoch in range(opt.epochs):

    for batch_step, data in enumerate(train_loader):
        global_steps = len(train_loader) * epoch + batch_step
        images, boxes, classes = data['img'], data['annot'][..., :4], data['annot'][..., -1]

        if global_steps < opt.warmup_steps:
            lr = float(global_steps / opt.warmup_steps * opt.lr)
            for param in optimizer.param_groups:
                param['lr'] = lr
        if epoch == 2:
            lr = opt.lr * 0.1

        optimizer.zero_grad()
        cls_loss, reg_loss, total_loss = model([images.cuda(), boxes.cuda(), classes.cuda()])
        writer.add_scalar('lr', lr, global_step=global_steps)
        writer.add_scalar('cls_loss', cls_loss.mean(), global_step=global_steps)
        writer.add_scalar('reg_loss', reg_loss.mean(), global_step=global_steps)
        writer.add_scalar('total_loss', total_loss.mean(), global_step=global_steps)
        total_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        print("cls_loss:%.4f reg_loss:%.4f total_loss:%.4f" % (cls_loss.mean(), reg_loss.mean(), total_loss.mean()))
    if epoch > opt.save_epochs:
        torch.save(model.state_dict(), f'runs/{logpath}/epoch_{epoch + 1}.pth')
        config['epoch'] = epoch
        json_file.save(config, logpath)
