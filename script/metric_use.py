import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def compute_speed(u, v):
    return np.sqrt(u**2 + v**2)

def compute_csi(pred, true, threshold):
    pred_binary = (pred >= threshold).astype(int)
    true_binary = (true >= threshold).astype(int)

    hits = np.sum((pred_binary == 1) & (true_binary == 1))
    misses = np.sum((pred_binary == 0) & (true_binary == 1))
    false_alarms = np.sum((pred_binary == 1) & (true_binary == 0))

    denominator = hits + misses + false_alarms
    return hits / denominator if denominator > 0 else 0.0

def compute_hss(pred, true, threshold):
    pred_binary = (pred >= threshold).astype(int)
    true_binary = (true >= threshold).astype(int)

    a = np.sum((pred_binary == 1) & (true_binary == 1))  # hits
    b = np.sum((pred_binary == 1) & (true_binary == 0))  # false alarms
    c = np.sum((pred_binary == 0) & (true_binary == 1))  # misses
    d = np.sum((pred_binary == 0) & (true_binary == 0))  # correct negatives

    numerator = 2 * (a * d - b * c)
    denominator = (a + c)*(c + d) + (a + b)*(b + d)

    return numerator / denominator if denominator != 0 else 0.0

def compute_mae(pred, true):
    return np.mean(np.abs(pred - true))

def compute_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def compute_pcc(pred, true):
    pred = np.ravel(pred)
    true = np.ravel(true)

    pred_mean = np.mean(pred)
    true_mean = np.mean(true)

    numerator = np.sum((pred - pred_mean) * (true - true_mean))
    denominator = np.sqrt(np.sum((pred - pred_mean) ** 2) * np.sum((true - true_mean) ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator

def compute_r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_fss(pred, true, eps=1e-6):
    """
    Compute Fraction Skill Score (FSS) as defined in the paper.

    Args:
        pred (np.ndarray): Predicted array of shape (T_out, ...).
        true (np.ndarray): Ground truth array of the same shape.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: FSS score between 0 and 1.
    """
    assert pred.shape == true.shape, "Shape of prediction and ground truth must match."

    T_out = pred.shape[0]

    mse = np.sum((true - pred) ** 2) / T_out
    denom = (np.sum(true ** 2) + np.sum(pred ** 2)) / T_out

    fss = 1.0 - (mse / (denom + eps))
    return fss

def compute_smape(pred, true, eps=1e-6):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    denominator = (np.abs(pred) + np.abs(true)) + eps
    smape = 2.0 * np.abs(pred - true) / denominator
    return np.mean(smape)

