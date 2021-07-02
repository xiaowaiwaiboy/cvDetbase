from cnn_model.data import *
from torchvision import transforms
import torch.utils.data
import argparse
from tensorboardX import SummaryWriter
import yaml
from cnn_model.model import build_detector

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='number of total epochs')
parser.add_argument('--save_epochs', type=int, default=10, help='when to save weights')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--warmup_steps', type=int, default=500)
parser.add_argument('--optimizer', type=str, default='SGD')
opt = parser.parse_args()

config = yaml.load(open('./cnn_model/config/detector.yaml', 'r'), Loader=yaml.FullLoader)
model_cfg = config.get('detector')
data_cfg = config.get('dataset_train')
writer = SummaryWriter(logdir=f'runs/{model_cfg["cfg"]}_{data_cfg["cfg"]}'.lower())

model = build_detector(cfg=model_cfg)
model = torch.nn.DataParallel(model)
model = model.cuda().train()

train_dataset = build_dataset(cfg=data_cfg,
                              transform=transforms.Compose([Augmenter(), Resizer(img_sizes=data_cfg.get('img_sizes'))]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_cfg.get('batch_size'), collate_fn=collate_fn)

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)

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
        print(
            "cls_loss:%.4f reg_loss:%.4f total_loss:%.4f" % (cls_loss.mean(), reg_loss.mean(), total_loss.mean()))
    if epoch > opt.save_epochs:
        torch.save(model.state_dict(), f'checkpoint/{model_cfg["cfg"]}_{epoch + 1}.pth')
