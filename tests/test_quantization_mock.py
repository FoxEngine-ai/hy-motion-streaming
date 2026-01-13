
import os
import sys
import torch
from unittest.mock import MagicMock, patch

# Mock some dependencies to isolate HYTextModel testing if needed,
# but we want to test real imports if possible.
# Adding project root to path
sys.path.append(os.getcwd())

from hymotion.network.text_encoders.text_encoder import HYTextModel

def test_quantization_config():
    print("Testing quantization config setup...")
    
    # We can't easily load the full 8B model in this test environment if it's not present,
    # but we can check if BitsAndBytesConfig is created correctly.
    
    with patch('hymotion.network.text_encoders.text_encoder.BitsAndBytesConfig') as MockBnB:
        with patch('hymotion.network.text_encoders.text_encoder.AutoModelForCausalLM') as MockAutoModel:
            # Mock tokenizer
            with patch('hymotion.network.text_encoders.text_encoder.AutoTokenizer') as MockTokenizer:
                MockTokenizer.from_pretrained.return_value = MagicMock()
                MockAutoModel.from_pretrained.return_value = MagicMock() # Mock model
                MockAutoModel.from_pretrained.return_value.config.hidden_size = 4096
                
                # Test 4bit
                print("Initializing HYTextModel with quantization_mode='4bit'")
                model = HYTextModel(llm_type="qwen3", quantization_mode="4bit")
                
                # Verify BitsAndBytesConfig was called with correct args
                MockBnB.assert_called_with(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("✅ 4-bit config verified")
                
                # Test 8bit
                print("Initializing HYTextModel with quantization_mode='8bit'")
                model = HYTextModel(llm_type="qwen3", quantization_mode="8bit")
                
                # Verify BitsAndBytesConfig usage
                # Note: creating new instance might call it again
                call_args = MockBnB.call_args
                # We expect the last call (or one of them) to be for 8bit
                # Reset mock to be sure
                MockBnB.reset_mock()
                model = HYTextModel(llm_type="qwen3", quantization_mode="8bit")
                MockBnB.assert_called_with(load_in_8bit=True)
                print("✅ 8-bit config verified")

if __name__ == "__main__":
    try:
        test_quantization_config()
        print("🎉 Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
