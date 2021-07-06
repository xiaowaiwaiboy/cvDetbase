from CNN.data import *
import yaml
from torchvision import transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import cv2

config = yaml.load(open('../config/fcos_fpn_r50.yaml', 'r'), Loader=yaml.FullLoader)
data_cfg = config.get('dataset_eval')

dataset = build_dataset(data_cfg)
train_dataset = build_dataset(cfg=data_cfg, transform=transforms.Compose([Augmenter(),
                                                                          Resizer(img_sizes=data_cfg.get('img_sizes'))]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, len(train_dataset.CLASSES_NAME))]

for num, data in enumerate(train_loader):
    image = np.asarray(data['img'].squeeze(0).permute(1, 2, 0), dtype='uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = data['annot'][..., :4].squeeze(0)
    classes = data['annot'][..., -1].squeeze(0)
    img_id = data['id'][0][1]

    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(image, pt1, pt2, tuple(255 * j for j in list(colors[int(classes[i])][:3])))
        # cv2.rectangle(image, pt1, pt2, (255, 0, 0))
        cls = "%s" % (VOCDataset.CLASSES_NAME[int(classes[i])])
        cv2.putText(image, cls, (int(box[0]), int(box[1]) + 20), 0, 1,
                    tuple(255 * j for j in list(colors[int(classes[i])][:3])), 2)
    cv2.imshow(f'{img_id}', image)
    cv2.waitKey(0)
    cv2.destroyWindow(f'{img_id}')
