import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # input shape (批量大小*通道数, 高, 宽) ; input 中的每一个值都在 (0, 1) 之间
    # target shape (批量大小*通道数, 高, 宽) ; target 中的每个值都是 0 或 1
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first


    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # inter 是预测正确的像素点的数量(这个数量不一定是整数，而是一种概率上表示的数量的和)
    # sum(dim=(-1, -2)) 是将倒数第 1 个维度和倒数第 2 个维度的值累加并归一到一个值
    # (input * target) = TP (是目标值预测正确)
    inter = 2 * (input * target).sum(dim=sum_dim)

    # input 是网络的预测值 是介于 0 和 1 之间的数值
    # target 是真实值 只有 0 和 1
    # input.sum + target.sum = 2TP + FP + FN (2*是目标值预测正确 + 是目标值但预测错误 + 不是目标值但预测错误)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # torch.where(condition, x, y)
    # condition is true ==> 返回结果为 x
    # condition is false ==> 返回结果为 y
    # 这一步好像有点多余
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # epsilon 用来避免分母出现 0 的情况
    dice = (inter + epsilon) / (sets_sum + epsilon)

    # mean 取平均数
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    # flatten(0, 1) 将 (批量大小, 通道数, 高, 宽) 拉伸成 (批量大小*通道数, 高, 宽)
    # 优先将通道数维度的内容按照 通道数量 个数的(高, 宽)为一组拼接在一起
    # 通道数量个数据 | 通道数量个数据 | ... | ==> 共有 批量大小 个 (通道数量个数据)
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(inputs: Tensor, target: Tensor, multiclass: bool = True):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(inputs, target, reduce_batch_first=True)
