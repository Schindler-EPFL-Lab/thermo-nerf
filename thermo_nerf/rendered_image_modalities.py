from enum import Enum


class RenderedImageModality(Enum):
    RGB = "img"
    DEPTH = "depth"
    ACCUMULATION = "accumulation"
    THERMAL = "thermal"
    THERMAL_COMBINED = "thermal_combined"
