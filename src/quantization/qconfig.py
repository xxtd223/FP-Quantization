from typing import Any

def prepare_quantization_config(
    hadamard_group_size: int, 
    format: str,
    pseudoquantization: bool = False
) -> dict[str, Any]:
    if format in ["mxfp", "nvfp", "hif"]:
        forward_dtype_str = "mxfp4" if format == "hif" else f"{format}4"  # 李代桃僵
        return {
            "forward_dtype": forward_dtype_str,
            "backward_dtype": "bf16",
            "forward_method": "abs_max",
            "hadamard_group_size":hadamard_group_size,
            "modules_to_not_convert": ["lm_head"],
            "quant_method": "fp_quant",
            "store_master_weights": False,
            "pseudoquantization": pseudoquantization
        }
    else:
        raise ValueError(f"Invalid format: {format}")
