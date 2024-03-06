from medpy import metric
import numpy as np


def dice(pred_mask, true_mask):
    """
    计算单个图的 dice 得分, dice 得分越大越好
    0 <= dice <= 1
    该函数的 dice 计算方式与 transunet 代码中的计算方式不同
    """
    if pred_mask.sum() > 0 and true_mask.sum() > 0:
        # pred_mask.sum() > 0 表示模型确实预测了某些像素点是当前所选择的类别
        # true_mask.sum() > 0 表示图像中也确实存在某些像素点是属于当前所选择的类别
        return metric.binary.dc(pred_mask, true_mask)
    elif pred_mask.sum() > 0 and true_mask.sum() == 0:
        # pred_mask.sum() > 0 表示模型确实预测了某些像素点是当前所选择的类别
        # true_mask.sum() == 0 表示图像中的所有像素点均不属于当前所选择的类别
        # 交集为空，则得分为 0
        return 0
    elif pred_mask.sum() == 0 and true_mask.sum() > 0:
        # 预测中没有，但是实际是有的，交集为空，得分为 0
        return 0
    else:
        # 预测中没有，实际也没有，说明预测全部正确，不参与测评
        return 0
        # return 1


def hd95(pred_mask, true_mask):
    """与 transunet 源码保持一致"""
    if pred_mask.sum() > 0 and true_mask.sum() > 0:
        return metric.binary.hd95(pred_mask, true_mask)
    elif pred_mask.sum() > 0 and true_mask.sum() == 0:
        return 0
    else:
        return 0


def score(pred_mask, true_mask):
    return dice(pred_mask, true_mask), hd95(pred_mask, true_mask)


def calculate_metric_percase(pred, gt):
    """测评单个 slice 的预测得分
    其中 pred 和 gt 是 pred_mask 和 true_mask
    大小都是 512， 512 的 h, w

    并且值是布尔类型的
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def batch_mean_score(pred_masks, true_masks):
    """
    pred_masks and true_masks 的值都是 bool type
    pred_masks : batch_size, h, w
    true_masks : batch_size, h, w
    计算一批图像的 dice 和 hd95 得分的平均值
    """
    return calculate_metric_percase(pred_masks, true_masks)

    assert pred_masks.shape[0] == true_masks.shape[0], 'pred_masks.shape[0]  must be eq to true_masks.shape[0]'
    nums = pred_masks.shape[0]
    scores = [0, 0]
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        dice_score, hd_score = calculate_metric_percase(pred_mask, true_mask)
        scores[0] += dice_score
        scores[1] += hd_score

    mean_dice = scores[0] / nums
    mean_hd = scores[1] / nums

    return mean_dice, mean_hd


def mask_is_empty(mask):
    return True if mask.sum() == 0 else False


def rid_off_empty(pred_masks, true_masks):
    """
    pred_masks && true_masks 的类型为 numpy
    """
    out_preds = []
    out_trues = []

    for pred_mask, true_mask in zip(pred_masks, true_masks):
        if not mask_is_empty(true_mask):
            out_preds.append(pred_mask)
            out_trues.append(true_mask)

    out_preds = np.stack(out_preds)
    out_trues = np.stack(out_trues)

    if len(out_preds) == 0 or len(out_trues) == 0:
        return None, None

    return out_preds, out_trues
