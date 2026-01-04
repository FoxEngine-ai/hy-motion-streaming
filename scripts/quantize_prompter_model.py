"""
Script to pre-quantize the prompter model to 4-bit and save it.
This significantly speeds up subsequent loading times.

Usage:
    python scripts/quantize_prompter_model.py
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantize_and_save_model(
    model_path: str,
    output_path: str,
    quantization_bits: int = 4,
):
    """
    Quantize the prompter model and save it.
    
    Args:
        model_path: Path to the original model
        output_path: Path to save the quantized model
        quantization_bits: Number of bits for quantization (4 or 8)
    """
    print(f">>> Starting model quantization...")
    print(f">>> Source model: {model_path}")
    print(f">>> Output path: {output_path}")
    print(f">>> Quantization: {quantization_bits}-bit")
    
    start_time = time.time()
    
    # Step 1: Load tokenizer
    print(f"\n>>> Step 1/4: Loading tokenizer...")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer_time = time.time() - tokenizer_start
    print(f">>> Tokenizer loaded in {tokenizer_time:.2f} seconds")
    
    # Step 2: Configure quantization
    print(f"\n>>> Step 2/4: Configuring {quantization_bits}-bit quantization...")
    if quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {quantization_bits}")
    
    # Step 3: Load and quantize model
    print(f"\n>>> Step 3/4: Loading and quantizing model...")
    print(f">>> This may take several minutes...")
    model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model_time = time.time() - model_start
    print(f">>> Model quantized in {model_time:.2f} seconds")
    
    # Step 4: Save quantized model
    print(f"\n>>> Step 4/4: Saving quantized model...")
    save_start = time.time()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    save_time = time.time() - save_start
    print(f">>> Model saved in {save_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f">>> Quantization completed successfully!")
    print(f">>> Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f">>> Quantized model saved to: {output_path}")
    print(f">>> {'='*60}")
    
    # Print model info
    print(f"\n>>> Model Information:")
    print(f">>> Device: {next(model.parameters()).device}")
    print(f">>> Dtype: {next(model.parameters()).dtype}")
    print(f">>> Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def main():
    """Main function."""
    # Default paths
    default_model_path = "ckpts/Text2MotionPrompter"
    default_output_path = "ckpts/Text2MotionPrompter_4bit"
    
    # Check if model exists
    if not os.path.exists(default_model_path):
        print(f">>> [ERROR] Model not found at: {default_model_path}")
        print(f">>> Please ensure the prompter model is downloaded first.")
        return
    
    # Check if output already exists
    if os.path.exists(default_output_path):
        response = input(f">>> Quantized model already exists at {default_output_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print(">>> Aborting.")
            return
    
    # Quantize and save
    quantize_and_save_model(
        model_path=default_model_path,
        output_path=default_output_path,
        quantization_bits=4,
    )
    
    print(f"\n>>> Next steps:")
    print(f">>> 1. The quantized model is now available at: {default_output_path}")
    print(f">>> 2. The Gradio app will automatically use this quantized model")
    print(f">>> 3. Loading time should be significantly faster (seconds instead of minutes)")


if __name__ == "__main__":
    main()