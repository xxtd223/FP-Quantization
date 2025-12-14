from enum import Enum

'''
定义各种枚举类型，包括量化格式、量化粒度、确定最佳量化参数的策略、量化顺序、比例尺的精度
'''


class QuantizationFormat(str, Enum):
    """
    Enum storing quantization format options
    """
    INT = "int"
    FP = "fp"
    NVFP = "nvfp"
    MXFP = "mxfp"


# 量化粒度：定义量化参数（即比例尺s和零点z）的计算和应用范围（粒度）
class QuantizationGranularity(str, Enum):
    """
    Enum storing quantization granularity options
    """
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"

class QuantizationObserver(str, Enum):
    """
    Enum storing quantization observer options
    """
    MINMAX = "minmax"
    MSE = "mse"

class QuantizationOrder(str, Enum):
    """
    Enum storing quantization order options
    """
    DEFAULT = "default"
    ACTIVATION = "activation"

class ScalePrecision(str, Enum):
    """
    Enum scale precision options
    """
    FP16 = "fp16"
    E4M3 = "e4m3"
    E8M0 = "e8m0"
