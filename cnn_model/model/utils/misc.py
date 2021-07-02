from functools import partial
import torch


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def calc_iou(box1, box2):
    """
    box1:(M,4)
    box2:(N,4)
    """
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # (M,N,2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (M,N,2)
    wh = torch.clamp(rb - lt, min=0.0)  # (M, N, 2)
    inter_area = wh[..., 0] * wh[..., 1]  # (M, N)
    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (M,)
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (N,)

    iou = inter_area / (area_box1[:, None] + area_box2 - inter_area + 1e-16)  # (M,N)

    return iou
