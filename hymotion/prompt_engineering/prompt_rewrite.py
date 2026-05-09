# prompt_rewrite.py
import base64
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from requests import exceptions as req_exc
from transformers import AutoModelForCausalLM, AutoTokenizer

# `from openai import OpenAI` is imported lazily inside OpenAIChatApi.__init__ —
# users running with disable_prompt_engineering=True (or anyone who never
# constructs an OpenAIChatApi) shouldn't have to install the openai package.

from .model_constants import REWRITE_AND_INFER_TIME_PROMPT_FORMAT

# logging.basicConfig(level=logging.INFO)


@dataclass
class ApiConfig:
    host: str
    user: str
    apikey: str
    model: str
    api_version: Optional[str] = None
    timeout: int = 3600
    source: str = "hymotion"


@dataclass
class RetryConfig:
    max_retries: int = 20
    base_delay: float = 1.0
    timeout: float = 30.0
    retry_status: Tuple[int, ...] = (429, 500, 502, 503, 504)
    max_delay: float = 1.0


class ApiError(Exception):
    pass


class ResponseParseError(Exception):
    pass


class OpenAIChatApi:
    def __init__(self, config: ApiConfig) -> None:
        # Lazy import — see top-of-file comment.
        from openai import OpenAI

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.client = OpenAI(
            api_key=self.config.apikey,
            base_url=self.config.host,
        )

    def call_data_eval(self, data: Union[str, Dict[str, Any]]):
        if isinstance(data, dict) and "messages" in data:
            raw_msgs = data["messages"]
            messages: List[Dict[str, str]] = []
            for m in raw_msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict) and ("text" in p):
                            parts.append(str(p.get("text", "")))
                    content = " ".join([t for t in parts if t])
                elif not isinstance(content, str):
                    content = str(content)
                messages.append({"role": role, "content": content})
            payload = {"model": self.config.model, "messages": messages}
            for k in (
                "temperature",
                "top_p",
                "max_tokens",
                "n",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "user",
            ):
                if k in data:
                    payload[k] = data[k]
        else:
            payload = {"model": self.config.model, "messages": [{"role": "user", "content": str(data)}]}
        try:
            resp = self.client.chat.completions.create(**payload)
            return resp
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise ApiError(f"OpenAI API call failed: {e}") from e


class ResponseParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def call_data_eval_with_retry(
        self, api: Union[OpenAIChatApi], data: str, retry_config: Optional[RetryConfig] = None
    ) -> Tuple[Union[Dict[str, Any], int], float, float]:
        if retry_config is None:
            retry_config = RetryConfig()

        last_error = None
        for attempt in range(retry_config.max_retries):
            start_time = time.time()
            cost = 0.0

            try:
                result = self._execute_request(api, data)
                end_time = time.time()
                parsed_result = self._parse_answer(result)
                self._validate_result(parsed_result)
                return parsed_result, cost, end_time - start_time

            except (
                concurrent.futures.TimeoutError,
                req_exc.RequestException,
                json.JSONDecodeError,
                ValueError,
                TypeError,
                ResponseParseError,
            ) as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if isinstance(e, req_exc.RequestException) and hasattr(e, "response"):
                    if e.response is not None and e.response.status_code not in retry_config.retry_status:
                        raise ApiError(f"Non-retryable error: {e.response.status_code}") from e
                if attempt < retry_config.max_retries - 1:
                    delay = self._calculate_delay(attempt, retry_config)
                    self.logger.info(f"JSON parsing failed, {delay:.1f} seconds later retry...")
                    time.sleep(delay)

        raise ApiError(f"Retry {retry_config.max_retries} times but still failed") from last_error

    def _execute_request(self, api: Union[OpenAIChatApi], data: str) -> Dict[str, Any]:
        response = api.call_data_eval(data)

        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if isinstance(response, dict):
                return response
            if hasattr(response, "__dict__"):
                return json.loads(json.dumps(response.__dict__, default=str))
        except Exception as e:
            raise ResponseParseError(f"Unable to parse OpenAI returned object: {type(response)} - {e}") from e

        raise ResponseParseError(f"Unknown response type: {type(response)}")

    def _extract_cost(self, payload: Dict[str, Any]) -> float:
        try:
            return float(payload.get("cost_info", {}).get("cost", 0)) / 1e6
        except (AttributeError, KeyError):
            return 0.0

    def _validate_result(self, result: Union[Dict[str, Any], int]) -> None:
        if isinstance(result, int):
            return
        elif isinstance(result, dict):
            required_fields = ["duration", "short_caption"]
            for field in required_fields:
                if not isinstance(result.get(field), (int, str)):
                    raise ResponseParseError(f"LLM returned invalid format: {field}")
        else:
            raise ResponseParseError(f"Unsupported answer type: {type(result)}")

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        delay = config.base_delay * (2**attempt) * (0.5 + random.random())
        return min(delay, config.max_delay)

    def _parse_answer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload, dict) and "choices" in payload:
            return self._parse_from_choices_field(payload)

        raise ResponseParseError("Unknown response format: expected choices")

    def _parse_from_choices_field(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        choices = payload.get("choices") or []
        if not choices:
            raise ResponseParseError("OpenAI returned empty")

        content = self._extract_content_from_choice(choices[0])

        if not isinstance(content, str) or not content.strip():
            raise ResponseParseError("OpenAI returned no valid content")

        return self._parse_json_content(content)

    def _extract_content_from_choice(self, choice: Any) -> Optional[str]:
        content = None

        if isinstance(choice, dict):
            # Try message content first
            msg = choice.get("message") or {}
            content = msg.get("content")
            # Fallback to delta content or text
            if content is None:
                delta = choice.get("delta") or {}
                content = delta.get("content", choice.get("text"))
        else:
            # Handle object-like choice (e.g. Pydantic model)
            msg = getattr(choice, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)

            if content is None:
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    content = getattr(delta, "content", None)

            if content is None:
                content = getattr(choice, "text", None)

        return content

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        cleaned = self._cleanup_fenced_json(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed, original content: {cleaned[:500]}...")
            raise ResponseParseError(f"JSON parsing failed: {e}") from e

    def _cleanup_fenced_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        if not text.lstrip().startswith("{") and "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                text = text[start : end + 1]
        return text


class PromptRewriter:
    def __init__(
        self, host: Optional[str] = None, model_path: Optional[str] = None, parser: Optional[ResponseParser] = None, lazy_load: bool = True
    ):
        self.parser = parser or ResponseParser()
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.lazy_load = lazy_load
        
        if host:
            self.api = OpenAIChatApi(
                ApiConfig(
                    host=host,
                    user="",
                    apikey="EMPTY",
                    model="Qwen3-30B-A3B-SFT",
                    api_version="",
                )
            )
        else:
            # Check for pre-quantized model first
            default_model_path = model_path or "Text2MotionPrompter/Text2MotionPrompter"
            quantized_model_path = default_model_path + "_4bit"
            
            # Use quantized model if it exists
            if os.path.exists(quantized_model_path):
                print(f">>> [INFO] Found pre-quantized model at: {quantized_model_path}")
                print(f">>> [INFO] Using quantized model for faster loading...")
                self.model_path = quantized_model_path
            else:
                print(f">>> [INFO] No pre-quantized model found at: {quantized_model_path}")
                print(f">>> [INFO] Using original model at: {default_model_path}")
                print(f">>> [INFO] To create a quantized model, run: python scripts/quantize_prompter_model.py")
                self.model_path = default_model_path
            
            self.tokenizer = None
            self.model = None
            
            # Only load model immediately if lazy_load is False
            if not lazy_load:
                self._load_model()
            else:
                print(f">>> [INFO] Lazy loading enabled - model will be loaded on first use")

    def _load_model(self):
        if self.model is None:
            import time
            print(f">>> Loading prompter model from {self.model_path}")
            print(f">>> [DEBUG] Starting model loading process...")
            
            start_time = time.time()
            
            # Step 1: Load tokenizer
            print(f">>> [DEBUG] Step 1/3: Loading tokenizer...")
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            tokenizer_time = time.time() - tokenizer_start
            print(f">>> [DEBUG] Tokenizer loaded in {tokenizer_time:.2f} seconds")
            
            # Step 2: Load model
            print(f">>> [DEBUG] Step 2/3: Loading model...")
            
            # Check if this is a pre-quantized model
            is_quantized = "_4bit" in self.model_path or "_8bit" in self.model_path
            
            # Determine target device and check available memory
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                
                # Use GPU 1 if available (GPU 0 is used by HY-Motion), otherwise use GPU 0
                if num_gpus > 1:
                    target_gpu = 1
                    print(f">>> [DEBUG] Multiple GPUs detected ({num_gpus}). Using GPU {target_gpu} for prompter model.")
                else:
                    target_gpu = 0
                    print(f">>> [DEBUG] Single GPU detected. Using GPU {target_gpu} for prompter model.")
                
                # Check available GPU memory on target GPU
                total_memory = torch.cuda.get_device_properties(target_gpu).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(target_gpu) / 1024**3
                free_memory = total_memory - allocated_memory
                
                print(f">>> [DEBUG] GPU {target_gpu} Memory: {total_memory:.2f} GB total, {allocated_memory:.2f} GB allocated, {free_memory:.2f} GB free")
                
                # If less than 10 GB free, use CPU offloading
                if free_memory < 10:
                    print(f">>> [WARNING] Low GPU memory ({free_memory:.2f} GB free). Using CPU offloading for prompter model.")
                    target_device = "cpu"
                    device_map = {"": "cpu"}
                else:
                    target_device = f"cuda:{target_gpu}"
                    print(f">>> [DEBUG] Target device: {target_device}")
                    # Use explicit device map to skip slow auto-detection
                    device_map = {"": target_device}
            else:
                target_device = "cpu"
                print(f">>> [DEBUG] Target device: {target_device} (no CUDA available)")
                device_map = {"": target_device}
            
            if is_quantized:
                print(f">>> [DEBUG] Loading pre-quantized model (no on-the-fly quantization needed)...")
                print(f">>> [DEBUG] Model config: torch_dtype=float16, device_map={device_map}")
                if target_device == "cpu":
                    print(f">>> [INFO] Loading on CPU (slower but saves GPU memory)...")
                    print(f">>> [DEBUG] Loading checkpoint shards (this may take 1-2 minutes on CPU)...")
                else:
                    print(f">>> [DEBUG] Loading checkpoint shards (this may take 30-60 seconds)...")
                model_start = time.time()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
            else:
                print(f">>> [DEBUG] Loading model with 4-bit quantization...")
                print(f">>> [DEBUG] Model config: torch_dtype=float16, device_map={device_map}, load_in_4bit=True")
                if target_device == "cpu":
                    print(f">>> [INFO] Loading on CPU (slower but saves GPU memory)...")
                    print(f">>> [DEBUG] Loading checkpoint shards (this may take 1-2 minutes on CPU)...")
                else:
                    print(f">>> [DEBUG] Loading checkpoint shards (this may take 30-60 seconds)...")
                model_start = time.time()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    load_in_4bit=True,
                    low_cpu_mem_usage=True,
                )
            
            model_time = time.time() - model_start
            print(f">>> [DEBUG] Model loaded in {model_time:.2f} seconds")
            
            # Step 3: Set to eval mode
            print(f">>> [DEBUG] Step 3/3: Setting model to eval mode...")
            eval_start = time.time()
            self.model.eval()
            eval_time = time.time() - eval_start
            print(f">>> [DEBUG] Model set to eval mode in {eval_time:.2f} seconds")
            
            total_time = time.time() - start_time
            print(f">>> [DEBUG] Total model loading time: {total_time:.2f} seconds")
            print(f">>> [DEBUG] Model device: {next(self.model.parameters()).device}")
            print(f">>> [DEBUG] Model dtype: {next(self.model.parameters()).dtype}")
            print(f">>> [DEBUG] Prompter model loading completed successfully!")

    def rewrite_prompt_and_infer_time(
        self,
        text: str,
        prompt_format: str = REWRITE_AND_INFER_TIME_PROMPT_FORMAT,
        retry_config: Optional[RetryConfig] = None,
    ) -> Tuple[float, str]:
        if self.host:
            self.logger.info("Start rewriting prompt...")
            try:
                result, cost, elapsed = self.parser.call_data_eval_with_retry(
                    self.api, prompt_format.format(text), retry_config
                )
                self.logger.info(f"Rewriting completed - cost: {cost:.6f}, time: {elapsed:.2f}s")
                return round(float(result["duration"]) / 30.0, 2), result["short_caption"]

            except Exception as e:
                self.logger.error(f"Prompt rewriting failed: {e}")
                raise
        else:
            # Lazy load model if not already loaded
            if self.model is None:
                print(f">>> [INFO] Loading prompter model on first use...")
                self._load_model()
            
            messages = [{"role": "user", "content": prompt_format.format(text)}]
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([full_prompt], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8192)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :].tolist(), skip_special_tokens=True)

            try:
                json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
                result = json.loads(json_str)
                return round(float(result["duration"]) / 30.0, 2), result["short_caption"]
            except:
                return 5.0, text


if __name__ == "__main__":
    # python -m hymotion.prompt_engineering.prompt_rewrite

    logging.basicConfig(level=logging.INFO)
    text = "person jumps after they runs"
    prompt_rewriter = PromptRewriter()
    result = prompt_rewriter.rewrite_prompt_and_infer_time(text)
    print(result)
