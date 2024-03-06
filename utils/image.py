import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom as scipy_zoom


def show_img(img):
    """输入一个 2d 的图像矩阵"""
    plt.imshow(img)
    plt.show()


def random_rotate(image, mask):
    angle = np.random.randint(-90, 90)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return image, mask


def random_flip(image, mask):
    axis = random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return image, mask


def zoom(image, mask, shape):
    """
    将图像和掩码的大小转化为 shape
    image: 图像
    mask: 掩码
    shape: output width, output height
    """
    image = scipy_zoom(image, (shape[0] / image.shape[0], shape[1] / image.shape[1]), order=3)
    mask = scipy_zoom(mask, (shape[0] / mask.shape[0], shape[1] / mask.shape[1]), order=0)

    return image, mask


def reshape(obj, shape, is_img=True):
    device = None

    if isinstance(obj, torch.Tensor):
        device = obj.device
        obj = obj.to(torch.device('cpu')).numpy()

    shape = (shape[0] / obj.shape[0], shape[1] / obj.shape[1])

    if is_img:
        outputs = ndimage.zoom(obj, shape, order=3)
    else:
        outputs = ndimage.zoom(obj, shape, order=0)

    if device is not None:
        outputs = torch.tensor(outputs, device=device)

    return outputs


def batch_reshape(obj, shape, is_img=True):
    """
    shape (h, w)
    """
    assert len(shape) == 2, 'shape should be (h, w)'
    device = None

    if isinstance(obj, torch.Tensor):
        device = obj.device
        obj = obj.to(torch.device('cpu')).numpy()

    # shape (..., h, w)
    shape = obj.shape[:-2] + (shape[0], shape[1])

    # shape 缩放因子
    shape = [target / cur for target, cur in zip(shape, obj.shape)]

    if is_img:
        outputs = ndimage.zoom(obj, shape, order=3)
    else:
        outputs = ndimage.zoom(obj, shape, order=0)

    if device is not None:
        outputs = torch.tensor(outputs, device=device)

    return outputs
