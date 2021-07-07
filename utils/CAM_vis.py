import numpy as np

from CNN.model import build_backbone, build_neck, build_head, build_detector
from CNN.data import build_dataset, Resizer, Augmenter, collate_fn, Normalizer
import torch.utils.data
import yaml
import torch
from torchvision import transforms

config = yaml.load(open('../config/fcos_fpn_r50.yaml', 'r'), Loader=yaml.FullLoader)
model_cfg = config.get('detector')
data_cfg = config.get('dataset_eval')
dataset = build_dataset(data_cfg)
train_dataset = build_dataset(cfg=data_cfg,
                              transform=transforms.Compose([Augmenter(),
                                                            Normalizer(),
                                                            Resizer(img_sizes=data_cfg.get('img_sizes'))]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)

model = build_detector(model_cfg)
model = torch.nn.DataParallel(model).cuda()


def deprocess_img(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x.cpu().detach().numpy(), 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def CAM_(featuremap):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    # start_t=time.time()
    B, C, H, W = featuremap.shape
    heat_map = torch.clamp_min(featuremap, 0)

    # heat_map -= heat_map.mean()
    # heat_map /= (heat_map.std() + 1e-5)
    # heat_map *= 0.1
    # heatmap = heat_map.data.cpu().numpy()
    # heatmap = heatmap.squeeze(0)
    # heatmap = np.uint8(255 * heatmap)
    # plt.imshow(heatmap)
    # plt.show()

    heat_map = torch.mean(heat_map, dim=1)
    max = heat_map.reshape(B, H * W).max(dim=1).values
    heat_map /= max.view(B, 1, 1)

    # heatmap vis
    heatmap = heat_map.data.cpu().numpy()
    heatmap = heatmap.squeeze(0)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (480, 640))
    cv2.imshow('hm', heatmap)
    cv2.waitKey(0)

    # a = torch.ones_like(heat_map)
    # b = torch.ones_like(heat_map) * -1
    # mask = torch.where(heat_map > heat_map.mean(), a, b)
    #
    # # # mask vis
    # # heat_map = mask.data.cpu().numpy()
    # # heat_map = heat_map.squeeze(0)
    # # heatmap = np.uint8(255 * heat_map)
    # # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # # cv2.imshow('mask',heatmap)
    # # cv2.waitKey(0)
    #
    # # mask = mask.unsqueeze(1)
    # mask = mask.unsqueeze(1).repeat(1, 9, 1, 1)
    # # mask = mask.permute(0, 2, 3, 1).contiguous()
    # mask = mask.permute(0, 2, 3, 1).contiguous().view(B, H * W * 9, -1)
    #
    # # end_t=time.time()
    # # cost_t=1000*(end_t-start_t)
    # # print("===>success processing mask, cost time %.2f ms"%cost_t)


for step, data in enumerate(train_loader):
    img, boxes, classes = data['img'], data['annot'][..., 4], data['annot'][..., -1]

    backbone = build_backbone(model_cfg.get('encoder').get('backbone_')).cuda()
    backbone_out = backbone(img.cuda())
    for feat in backbone_out:
        CAM_(feat)

    neck = build_neck(model_cfg.get('encoder').get('neck_')).cuda()
    neck_out = neck(backbone_out)
    for feat in neck_out:
        CAM_(feat)
