import torch


def mae_thermal(gt, pred, threshold, cold_flag, max_temperature, min_temperature):
    if cold_flag:
        indices_foreground = torch.where(gt < threshold)
    else:
        indices_foreground = torch.where(gt > threshold)

    gt = gt[indices_foreground]
    gt = gt * (max_temperature - min_temperature) + min_temperature

    pred = pred[indices_foreground]
    pred = pred * (max_temperature - min_temperature) + min_temperature
    mae = torch.abs(gt - pred)

    return mae
