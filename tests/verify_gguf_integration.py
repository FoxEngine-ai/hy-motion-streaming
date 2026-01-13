
import os
import sys
import torch
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from hymotion.network.text_encoders.text_encoder import HYTextModel
from hymotion.utils.t2m_runtime import T2MRuntime


# Define a mock class that behaves like the real one for isinstance checks
class MockLlamaClass:
    def __init__(self, *args, **kwargs):
        self.embed = MagicMock()
        self.tokenize = MagicMock()
        self.n_ctx = MagicMock(return_value=2048)
        self.n_embd = MagicMock(return_value=4096)

class TestGGUFIntegration(unittest.TestCase):
    
    @patch('hymotion.network.text_encoders.text_encoder.Llama', new=MockLlamaClass)
    @patch('hymotion.network.text_encoders.text_encoder.AutoTokenizer')
    @patch('os.walk')
    @patch('hymotion.network.text_encoders.text_encoder.AutoModelForCausalLM')
    def test_hytextmodel_gguf_init(self, MockAutoModel, MockWalk, MockTokenizer):
        print("\n[Test 1] Testing HYTextModel GGUF Initialization...")
        
        # Setup mocks
        MockWalk.return_value = [
            ('/mock/path', [], ['model-q4_k_m.gguf', 'config.json'])
        ]
        
        mock_tokenizer_instance = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Initialize
        model = HYTextModel(
            llm_type="qwen3",
            use_gguf=True
        )
        
        # Verify Llama was used (by checking if instance is our MockLlamaClass)
        self.assertIsInstance(model.llm_text_encoder, MockLlamaClass)
        
        # Check model path arg
        # We can't check init args easily with a class replacement unless we wrap init, 
        # but the fact it initialized successfully implies it found the file logic.
        print("✅ HYTextModel correctly initialized Llama (MockClass)")

        
        # Verify AutoModel (standard path) was NOT called
        MockAutoModel.from_pretrained.assert_not_called()
        print("✅ AutoModel skipped")

    @patch('hymotion.network.text_encoders.text_encoder.Llama', new=MockLlamaClass)
    @patch('hymotion.network.text_encoders.text_encoder.AutoTokenizer')
    @patch('os.walk')
    def test_encode_llm_monkeypatch(self, MockWalk, MockTokenizer):
        print("\n[Test 2] Testing encode_llm Monkeypatch Logic...")
        
        # Setup mocks
        MockWalk.return_value = [('/mock/path', [], ['model.gguf'])]
        
        # Mock Tokenizer
        mock_hf_tokenizer = MagicMock()
        MockTokenizer.from_pretrained.return_value = mock_hf_tokenizer
        
        # Simulate tokenizer call
        mock_encoding = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_hf_tokenizer.return_value = mock_encoding
        mock_hf_tokenizer.apply_chat_template.return_value = "mock_prompt"
        
        # Init Model (will create MockLlamaClass instance)
        model = HYTextModel(llm_type="qwen3", use_gguf=True)
        model.llm_tokenizer = mock_hf_tokenizer
        
        # Setup embed return output on the INSTANCE
        # model.llm_text_encoder is the instance of MockLlamaClass
        # We need to set the return value of its embed method
        
        # Note: Llama.embed returns list of lists [Seq, Dim]
        # n_embd is 4096
        model.llm_text_encoder.embed.return_value = [[0.1]*4096] * 3 # 3 tokens, dim 4096
        
        # Call encode_llm
        text = ["Test prompt"]
        model.encode_llm(text)
        
        # Verify embed was called
        model.llm_text_encoder.embed.assert_called()
        print(f"Llama.embed called with: {model.llm_text_encoder.embed.call_args}")
        
        # Verify inputs were the dummy string (because we monkeypatch tokenize)
        args, _ = model.llm_text_encoder.embed.call_args
        self.assertEqual(args[0], "mock_string_to_trigger_mock_tokenize")
        
        print("✅ encode_llm ran successfully (mocks verified)")


if __name__ == "__main__":
    unittest.main()
