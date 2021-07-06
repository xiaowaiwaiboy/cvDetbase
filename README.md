# cvDetbase
&emsp;Organize object detection models,divide the model into backbone network, neck, head...  


&emsp;By modifying the YAML configuration file,developer can easily combine different model configurations, such as changing different backbone networks, necks, etc.  


Some things to note are:   
- The parameter `out_indices` of the backbone network configuration must match the input of neck.  
- This base only integrates excellent open source detection projects for ease of use, so the code style of each object detection model may be completely different.


Supported now:  
  
|  backbone   | neck  | head       | loss       | optimizer |detector  | dataset |
|  :----:  | :----:   | :---:  |  :---: | :---: | :---:     |:---: |
| resnet      | fpn   | retinahead | focal_loss | SGD    |retinanet | voc |
| mobilenet   | bifpn | fcoshead   | smooth_l1  | Adam    |fcos      | coco|
| darknet |           | efficientdet|           | Adamw   |efficientdet      
| densenet |  
| efficientnet |
| resNext |
| vgg |
| inception_v3|

Train:  
1. Modify the parameters `epochs`, `n_gpu`... in the `train.py` file.
2. Select the configuration file and modify the relevant parameters, then run `train.py`.
3. `.pth` and log file will be saved at `runs/***/{timestamp}/***`.

Validate:  
1. Select the configuration file and modify the relevant parameters, then run `test.py`.




## Reference:
- [https://github.com/EasonCai-Dev/torch_backbones](https://github.com/EasonCai-Dev/torch_backbones)
- [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- [https://github.com/zhenghao977/RetinaNet-Pytorch-36.4AP](https://github.com/zhenghao977/RetinaNet-Pytorch-36.4AP)
- [https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
