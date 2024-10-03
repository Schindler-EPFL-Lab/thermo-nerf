import torch
from torch import Tensor


def mae_thermal(
    gt: Tensor,
    pred: Tensor,
    cold_flag: bool,
    max_temperature: float,
    min_temperature: float,
    threshold: float | None = None,
) -> Tensor:
    """
    Calculate the thermal mean absolute error between `gt` and `pref`.

    Project `gt` and `pref` back to the range of `min_temperature` to `max_temperature`.
    `cold_flag` is set to true when the temperature of the temperature is lower than
    environment. `threshold` is a mask value to decide the region of interest.

    :returns: mean absolute error
    """
    if threshold:
        if cold_flag:
            indices_foreground = torch.where(gt < threshold)
        else:
            indices_foreground = torch.where(gt > threshold)
        gt = gt[indices_foreground]
        pred = pred[indices_foreground]

    gt = gt * (max_temperature - min_temperature) + min_temperature
    pred = pred * (max_temperature - min_temperature) + min_temperature
    mae = torch.mean(torch.abs(gt - pred))

    return mae
