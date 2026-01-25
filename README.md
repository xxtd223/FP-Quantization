# A Study on the Performance of Different Quantification Formats in LLM Models

### Quantization
---
An example of the quantization script:

```shell
#!/bin/bash

if [ "$#" -h 3 ]; then
    echo "用法: $0 <模型路径> <数据类型> <变换策略>"
    echo "数据类型: int, fp, mxfp, nvfp"
    echo "变换策略: identity, hadamard, dct, dst, fast_food, gsr"
    exit 1
fi

MODEL=$1
FORMAT=$2
TRANSFORM_CLASS=$3

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

if [[ "$FORMAT" == "int" || "$FORMAT" == "fp" ]]; then
    EXPORT_QUANTIZATION=""
    EVAL_OPENLLM=1
else
    EXPORT_QUANTIZATION="pseudoquant"
    EVAL_OPENLLM=1
fi

EVAL_PERPLEXITY=1
GPTQ=0  # 开启 GPTQ
W_BITS=4 # 权重值
A_BITS=4 # 激活值
LM_EVAL_BATCH_SIZE="16"

SCRIPT_ARGS=""

if [[ $GPTQ == 1 ]]; then SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"; fi
if [[ $EVAL_PERPLEXITY == 1 ]]; then SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity"; fi
if [[ $EVAL_OPENLLM == 1 ]]; then 
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_openllm" 
fi

MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="eval_results/${MODEL_ID}-${FORMAT}-${TRANSFORM_CLASS}-${TIMESTAMP}"

if [[ -n "$EXPORT_QUANTIZATION" ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --export_quantized_model ${EXPORT_QUANTIZATION}"
fi

# 评估
echo "--------------------------------------------------"
echo "🚀 启动实验"
echo "模型: $MODEL"
echo "格式: $FORMAT"
echo "变换: $TRANSFORM_CLASS"
echo "保存路径: $SAVE_DIR"
echo "--------------------------------------------------"

python FP-Quantization/model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=${FORMAT} \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --w_group_size=32 \
    --a_group_size=32 \
    --transform_class=${TRANSFORM_CLASS} \
    --hadamard_group_size=128 \
    --dataset_name_or_path=fineweb-edu \
    --num_sequences=128 \
    --sequence_length=2048 \
    --dtype="auto" \
    --lm_eval_batch_size=${LM_EVAL_BATCH_SIZE} \
    --save_path "$SAVE_DIR" \
    --cpu_offload_activations \
    --cpu_offload_modules \
    --fuse_global_scale \
    --amp \
    $SCRIPT_ARGS

echo "✅ 实验任务执行完毕！"
```

Above:
* `--model_name_or_path` - The model to quantize. (Llama and Qwen3 models are supported)
* `--format` - The quantization format (int, fp, mxfp, nvfp, hif). 
* `--w_bits` - The number of bits to quantize the weights to.
* `--a_bits` - The number of bits to quantize the activations to.
* `--w_group_size` - The number of weights to quantize together.
* `--a_group_size` - The number of activations to quantize together.
* `--init` - Transform initialization.
* `--transform_class` - Transform class. We provide the following options:
    * `identity` - Identity transform
    * `hadamard` - Hadamard transform
    * `dct` - Discrete cosine transform
    * `dst` - Discrete sine transform
    * `fast_food` - Fast food transform
    * `gsr` - Grouped sequency aligned transform
* `--hadamard_group_size` - Transform group size.
* `--dataset_name_or_path` - Dataset to use for calibration.
* `--sequence_length` - Calibration sequence length.
* `--dtype` - Data type to load the model.
* `--amp` - Whether to use automatic mixed precision.
* `--export_quantized_model` - Whether to export quantized model in `realquant` or `pseudoquant` format. The former allows one to run quantized model with the help of [QuTLASS](https://github.com/IST-DASLab/qutlass) integration, while the latter produces fake quantized model runnable with `triton` kernels.

For evaluation, we provide the following options:

* `--eval_perplexity` - Whether to evaluate perplexity after quantization.
* `--eval_openllm` - Whether to evaluate OpenLLM v1 openllm after quantization.
* `--lm_eval_batch_size` - LM eval batch size to evaluate after quantization.
* `--fuse_global_scale` - Whether to fuse global scale in qkv and gate_up projections as required by `vLLM`.
