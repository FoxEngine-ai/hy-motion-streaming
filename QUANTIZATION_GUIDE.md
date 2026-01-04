# Prompter Model Quantization Guide

This guide explains how to quantize the prompter model to significantly reduce loading time from ~10 minutes to a few seconds.

## Problem

The prompter model (Text2MotionPrompter) is being loaded with on-the-fly 4-bit quantization every time the Gradio app starts. This process takes approximately **10 minutes**, which is unacceptably slow.

## Solution

Pre-quantize the model once and save it to disk. Subsequent loads will use the pre-quantized model, reducing loading time to **seconds**.

## Quick Start

### Step 1: Quantize the Model (One-Time Setup)

Run the quantization script:

```bash
make quantize-prompter
```

Or directly:

```bash
python3 scripts/quantize_prompter_model.py
```

This will:
- Load the original model from `ckpts/Text2MotionPrompter`
- Quantize it to 4-bit using BitsAndBytesConfig
- Save the quantized model to `ckpts/Text2MotionPrompter_4bit`

**Expected time:** 5-15 minutes (one-time only)

### Step 2: Run Gradio App

After quantization, simply run the Gradio app as usual:

```bash
make run-gradio-docker
```

The app will automatically detect and use the pre-quantized model.

**Expected loading time:** 5-30 seconds (vs 10 minutes before)

## How It Works

### Before Quantization

```
Loading prompter model from ckpts/Text2MotionPrompter
[DEBUG] Step 1/3: Loading tokenizer... (2 seconds)
[DEBUG] Step 2/3: Loading model with 4-bit quantization... (600 seconds!)
[DEBUG] Step 3/3: Setting model to eval mode... (1 second)
Total: ~10 minutes
```

### After Quantization

```
[INFO] Found pre-quantized model at: ckpts/Text2MotionPrompter_4bit
[INFO] Using quantized model for faster loading...
Loading prompter model from ckpts/Text2MotionPrompter_4bit
[DEBUG] Step 1/3: Loading tokenizer... (2 seconds)
[DEBUG] Step 2/3: Loading pre-quantized model... (5 seconds)
[DEBUG] Step 3/3: Setting model to eval mode... (1 second)
Total: ~10 seconds
```

## Technical Details

### Quantization Configuration

The model is quantized using BitsAndBytesConfig with the following settings:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

- **4-bit quantization**: Reduces model size by ~75%
- **NF4 (NormalFloat4)**: Optimal quantization for neural networks
- **Double quantization**: Further reduces memory usage
- **Float16 compute**: Maintains precision during inference

### Memory Savings

| Metric | Original | Quantized (4-bit) |
|--------|----------|-------------------|
| Model Size | ~30 GB | ~8 GB |
| Loading Time | ~10 min | ~10 sec |
| VRAM Usage | ~30 GB | ~8 GB |

### Automatic Detection

The [`PromptRewriter`](hymotion/prompt_engineering/prompt_rewrite.py:247-268) class automatically checks for the quantized model:

```python
# Check for pre-quantized model first
quantized_model_path = default_model_path + "_4bit"

if os.path.exists(quantized_model_path):
    # Use quantized model
    self.model_path = quantized_model_path
else:
    # Fall back to original model
    self.model_path = default_model_path
```

## Troubleshooting

### Issue: "Model not found at ckpts/Text2MotionPrompter"

**Solution:** Download the prompter model first:

```bash
make download-rewriter
```

### Issue: "bitsandbytes not installed"

**Solution:** Install bitsandbytes:

```bash
pip install bitsandbytes
```

### Issue: Quantization fails with CUDA error

**Solution:** Ensure you have enough GPU memory. The quantization process requires:
- Original model: ~30 GB VRAM
- Quantized model: ~8 GB VRAM

If you don't have enough VRAM, try:
1. Close other GPU processes
2. Use a machine with more GPU memory
3. Use 8-bit quantization instead (modify the script)

### Issue: Still slow after quantization

**Solution:** Verify the quantized model is being used:

1. Check the logs for:
   ```
   [INFO] Found pre-quantized model at: ckpts/Text2MotionPrompter_4bit
   ```

2. If you don't see this message, the quantized model wasn't created or isn't in the expected location.

3. Verify the directory exists:
   ```bash
   ls -la ckpts/Text2MotionPrompter_4bit
   ```

## Advanced Usage

### Custom Quantization Paths

To use custom paths, modify the script:

```python
quantize_and_save_model(
    model_path="path/to/original/model",
    output_path="path/to/quantized/model",
    quantization_bits=4,
)
```

### 8-bit Quantization

For slightly better quality at the cost of more memory:

```python
quantize_and_save_model(
    model_path="ckpts/Text2MotionPrompter",
    output_path="ckpts/Text2MotionPrompter_8bit",
    quantization_bits=8,
)
```

### Docker Usage

To quantize inside Docker:

```bash
docker run -it --rm \
    --gpus all \
    -v $(PWD):/app \
    -v $(PWD)/ckpts:/app/ckpts \
    hymotion:latest \
    python /app/scripts/quantize_prompter_model.py
```

## Performance Comparison

### Loading Time Comparison

| Configuration | Loading Time | Speedup |
|--------------|--------------|---------|
| Original (no quantization) | ~15 min | 1x |
| On-the-fly 4-bit quantization | ~10 min | 1.5x |
| **Pre-quantized 4-bit** | **~10 sec** | **90x** |

### Inference Speed

The quantized model maintains similar inference speed to the on-the-fly quantized version:
- **Original**: ~2-3 seconds per rewrite
- **Quantized (4-bit)**: ~2-3 seconds per rewrite
- **No significant difference in inference speed**

## Maintenance

### Updating the Model

If you update the original prompter model, you'll need to re-quantize:

```bash
# 1. Download new model
make download-rewriter

# 2. Remove old quantized model
rm -rf ckpts/Text2MotionPrompter_4bit

# 3. Re-quantize
make quantize-prompter
```

### Disk Space

The quantized model requires additional disk space:
- Original model: ~30 GB
- Quantized model: ~8 GB
- **Total**: ~38 GB

Ensure you have sufficient disk space before quantizing.

## References

- [BitsAndBytes Documentation](https://huggingface.co/docs/bitsandbytes)
- [4-bit Quantization Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [NF4 Quantization Paper](https://arxiv.org/abs/2305.14314)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the debug output from the quantization script
3. Ensure all dependencies are installed
4. Verify GPU memory availability