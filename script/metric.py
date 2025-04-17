import numpy as np
import torch

def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero
    return torch.mean(loss)

def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = (prediction - target) ** 2  # Compute squared error
    loss *= mask  # Apply mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)  # Return the mean of the masked loss

def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))

def masked_r2(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    labels_mean = torch.sum(labels * mask) / torch.sum(mask)

    ss_total = torch.sum((labels - labels_mean) ** 2 * mask)
    ss_res = torch.sum((labels - preds) ** 2 * mask)

    r2 = 1 - ss_res / ss_total

    return r2

def masked_pearson_corr(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # 计算均值
    mean_preds = torch.sum(preds * mask) / torch.sum(mask)
    mean_labels = torch.sum(labels * mask) / torch.sum(mask)
    # 计算偏差
    deviation_preds = preds - mean_preds
    deviation_labels = labels - mean_labels
    # 计算皮尔逊相关系数
    numerator = torch.sum(deviation_preds * deviation_labels * mask)
    denominator = torch.sqrt(torch.sum(deviation_preds ** 2 * mask) * torch.sum(deviation_labels ** 2 * mask))
    pearson_corr = numerator / denominator
    return pearson_corr


def masked_mape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    # mask to exclude zero values in the target
    zero_mask = ~torch.isclose(target, torch.tensor(0.0).to(target.device), atol=5e-5)

    # mask to exclude null values in the target
    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    # combine zero and null masks
    mask = (zero_mask & null_mask).float()

    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs((prediction - target) / target)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)


def masked_smape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    # mask to exclude zero values in the target
    zero_mask = ~torch.isclose(target, torch.tensor(0.0).to(target.device), atol=5e-5)

    # mask to exclude null values in the target
    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    # combine zero and null masks
    mask = (zero_mask & null_mask).float()

    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs(prediction - target) / ((prediction.abs() + target.abs()) / 2)
    loss *= mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)

def masked_fss(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    计算带掩码的 Fuzzy Skill Score (FSS)
    :param prediction: 预测值张量
    :param target: 实际值张量
    :param null_val: 用于标记缺失值的值，默认为 np.nan
    :return: FSS 值
    """
    # 处理缺失值掩码
    if np.isnan(null_val):
        null_mask = ~torch.isnan(target)  # 如果实际值是 NaN，忽略这些值
    else:
        eps = 5e-5
        null_mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)  # 如果是指定的缺失值，忽略这些值

    # 掩码处理：计算 P_f 和 P_o 的平方差
    mask = null_mask.float()

    mask /= torch.mean(mask)  # 归一化掩码
    mask = torch.nan_to_num(mask)

    # 计算 FSS 公式中的分子：1/N * ∑(P_f - P_o)^2
    numerator = torch.sum((prediction - target) ** 2) / target.numel()

    # 计算 FSS 公式中的分母：1/N * [∑P_f^2 + ∑P_o^2]
    denominator = (torch.sum(prediction ** 2) + torch.sum(target ** 2)) / target.numel()

    # 计算 FSS 值
    fss = 1 - (numerator / denominator)

    # 处理缺失值掩码，忽略缺失值对 FSS 的影响
    fss *= torch.mean(mask)  # 用掩码权重来调整最终的 FSS 值

    return fss

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    pear=masked_pearson_corr(pred,real,0.0).item()
    r= masked_r2(pred,real,0.0).item()
    smape=masked_smape(pred,real,0.0).item()
    fss=masked_fss(pred,real,0.0).item()
    return mae, rmse, pear, r, smape, fss

