#!/usr/bin/env python3
"""
FastAPI Streaming Server for HY-Motion-1.0

REST API for generating motion from text prompts.
Returns JSON response with SMPL-H motion data.

Usage:
    python streaming_server.py
    # or with uvicorn:
    uvicorn streaming_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import json
import uuid
import threading
from typing import Optional, Union, Dict, Any, List
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the runtime and necessary utilities
from hymotion.utils.t2m_runtime import T2MRuntime
from hymotion.utils.visualize_mesh_web import save_visualization_data, get_output_dir
from hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask
from hymotion.utils.type_converter import get_module_device

# Global runtime instance
_runtime_instance: Optional['StreamingT2MRuntime'] = None
_runtime_lock = threading.Lock()


def _init_runtime_if_needed() -> 'StreamingT2MRuntime':
    """Initialize the runtime instance if not already created."""
    global _runtime_instance
    
    with _runtime_lock:
        if _runtime_instance is None:
            print(">>> Initializing StreamingT2MRuntime for FastAPI...")
            
            # Try multiple possible model paths
            possible_paths = [
                "ckpts/tencent/HY-Motion-1.0",
                "downloaded_models/HY-Motion-1.0-Lite",
                "downloaded_models/HY-Motion-1.0",
                "ckpts/HY-Motion-1.0-Lite",
            ]
            
            found_path = None
            for model_path in possible_paths:
                cfg = os.path.join(model_path, "config.yml")
                ckpt = os.path.join(model_path, "latest.ckpt")
                if os.path.exists(cfg) and os.path.exists(ckpt):
                    found_path = model_path
                    print(f">>> [INFO] Found model files in {model_path}")
                    break
            
            if found_path is None:
                found_path = "ckpts/tencent/HY-Motion-1.0"
                print(f">>> [WARNING] Model files not found, will attempt Hugging Face download")
            
            cfg = os.path.join(found_path, "config.yml")
            ckpt = os.path.join(found_path, "latest.ckpt")
            
            skip_model_loading = not (os.path.exists(cfg) and os.path.exists(ckpt))
            
            _runtime_instance = StreamingT2MRuntime(
                config_path=cfg,
                ckpt_name=ckpt,
                device_ids=[0],
                disable_prompt_engineering=True,
                skip_model_loading=skip_model_loading,
            )
    
    return _runtime_instance


class StreamingT2MRuntime(T2MRuntime):
    """Extended T2MRuntime for FastAPI server."""

    def __init__(
        self,
        config_path: str,
        ckpt_name: str = "latest.ckpt",
        skip_text: bool = False,
        device_ids: Union[list[int], None] = None,
        skip_model_loading: bool = False,
        force_cpu: bool = False,
        disable_prompt_engineering: bool = False,
        prompt_engineering_host: Optional[str] = None,
        prompt_engineering_model_path: Optional[str] = None,
    ):
        print(">>> [INFO] Loading model in FP32 on GPU 0")
        super().__init__(
            config_path=config_path,
            ckpt_name=ckpt_name,
            skip_text=skip_text,
            device_ids=device_ids,
            skip_model_loading=skip_model_loading,
            force_cpu=force_cpu,
            disable_prompt_engineering=disable_prompt_engineering,
            prompt_engineering_host=prompt_engineering_host,
            prompt_engineering_model_path=prompt_engineering_model_path,
        )
    
    def generate_motion(
        self,
        text: str,
        seed: int = 0,
        duration: float = 5.0,
        cfg_scale: float = 7.0,
    ) -> Dict[str, Any]:
        """
        Generate motion from text prompt.
        
        Args:
            text: Input text prompt
            seed: Random seed (0 for random)
            duration: Motion duration in seconds
            cfg_scale: Classifier-free guidance scale (1-10)
            
        Returns:
            Dictionary containing motion data and metadata
        """
        self.load()
        
        # Handle seed
        if seed == 0:
            seed = int(torch.randint(0, 2**31, (1,)).item())
        seeds = [seed]
        
        pi = self._acquire_pipeline()
        
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()
            
            device = get_module_device(pipeline)
            length = int(round(duration * pipeline.output_mesh_fps))
            
            # Clamp length
            if length > pipeline.train_frames or length < min(pipeline.train_frames, 20):
                length = min(length, pipeline.train_frames)
                length = max(length, min(pipeline.train_frames, 20))
            
            # Text encoding
            text_list = [text] * len(seeds)
            hidden_state_dict = pipeline.encode_text({"text": text_list})
            
            vtxt_input = hidden_state_dict["text_vec_raw"]
            ctxt_input = hidden_state_dict["text_ctxt_raw"]
            ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
            
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            x_length = torch.LongTensor([length] * len(seeds)).to(device)
            x_mask_temporal = length_to_mask(x_length, pipeline.train_frames)
            
            text_guidance_scale = cfg_scale
            do_classifier_free_guidance = text_guidance_scale > 1.0 and not pipeline.uncondition_mode
            
            if do_classifier_free_guidance:
                silent_text_feat = pipeline.null_vtxt_feat.expand(*vtxt_input.shape)
                vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)
                silent_ctxt_input = pipeline.null_ctxt_input.expand(*ctxt_input.shape)
                ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)
                ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
                x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
            
            def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
                x_pred = pipeline.motion_transformer(
                    x=x_input,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t.expand(x_input.shape[0]),
                    x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )
                if do_classifier_free_guidance:
                    x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                    x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)
                return x_pred
            
            from torchdiffeq import odeint
            
            t = torch.linspace(0, 1, pipeline.validation_steps + 1, device=device)
            y0 = pipeline.noise_from_seeds(
                torch.zeros(
                    1,
                    pipeline.train_frames,
                    pipeline._network_module_args["input_dim"],
                    device=device,
                ),
                seeds,
                random_generator_on_gpu=pipeline.random_generator_on_gpu,
            )
            
            print(f">>> Generating motion: '{text}' | {duration}s | seed={seed} | cfg={cfg_scale}")
            start_time = time.time()
            
            with torch.no_grad():
                trajectory = odeint(fn, y0, t, **pipeline._noise_scheduler_cfg)
            
            generation_time = time.time() - start_time
            
            # Get final output with smoothing
            sampled = trajectory[-1][:, :length, ...].clone()
            final_output = pipeline.decode_motion_from_latent(
                sampled,
                should_apply_smooothing=True
            )
            
            print(f">>> Generated {length} frames in {generation_time:.2f}s ({duration/generation_time:.2f}x realtime)")
            
            return {
                "motion_data": final_output,
                "generation_time": generation_time,
                "output_duration": duration,
                "frame_count": length,
                "seed": seed,
                "text": text,
                "cfg_scale": cfg_scale,
            }
            
        finally:
            self._release_pipeline(pi)


# ============ Pydantic Models ============

class GenerateRequest(BaseModel):
    """Request model for motion generation."""
    prompt: str = Field(..., description="Text prompt describing the motion")
    duration: float = Field(default=5.0, ge=1.0, le=10.0, description="Motion duration in seconds")
    seed: int = Field(default=0, ge=0, description="Random seed (0 for random)")
    cfg_scale: float = Field(default=7.0, ge=1.0, le=10.0, description="Classifier-free guidance scale")


class MotionFrame(BaseModel):
    """Single frame of motion data."""
    poses: List[float] = Field(..., description="Joint poses (156 values = 52 joints × 3 axis-angle)")
    trans: List[float] = Field(..., description="Root translation (3 values: x, y, z)")
    Rh: List[float] = Field(..., description="Root rotation (3 values: axis-angle)")


class GenerateResponse(BaseModel):
    """Response model for motion generation."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# ============ FastAPI App ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    print(">>> FastAPI server starting, initializing runtime...")
    try:
        _init_runtime_if_needed()
        print(">>> Runtime initialized successfully")
    except Exception as e:
        print(f">>> Warning: Failed to initialize runtime on startup: {e}")
    yield
    print(">>> FastAPI server shutting down")


app = FastAPI(
    title="HY-Motion Streaming Server",
    description="REST API for generating human motion from text prompts using HY-Motion-1.0",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "HY-Motion Streaming Server"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    runtime = _runtime_instance
    return {
        "status": "ok",
        "runtime_loaded": runtime is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_motion(request: GenerateRequest):
    """
    Generate motion from a text prompt.
    
    Returns JSON with SMPL-H motion data including:
    - poses: Joint rotations in axis-angle format (156 values per frame)
    - trans: Root translation (3 values per frame)
    - Rh: Root rotation (3 values per frame)
    - betas: Body shape parameters
    """
    try:
        runtime = _init_runtime_if_needed()
        
        result = runtime.generate_motion(
            text=request.prompt,
            seed=request.seed,
            duration=request.duration,
            cfg_scale=request.cfg_scale,
        )
        
        motion_data = result["motion_data"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save and get SMPL data
        memory_data, base_filename = save_visualization_data(
            output=motion_data,
            text=request.prompt,
            rewritten_text=request.prompt,
            timestamp=timestamp,
            output_dir=None,
        )
        
        # Convert to JSON-serializable format
        smpl_data = memory_data["smpl_data"][0]  # First batch item
        
        # Flatten arrays for JSON
        poses = smpl_data["poses"].flatten().tolist() if isinstance(smpl_data["poses"], np.ndarray) else smpl_data["poses"]
        trans = smpl_data["trans"].flatten().tolist() if isinstance(smpl_data["trans"], np.ndarray) else smpl_data["trans"]
        Rh = smpl_data["Rh"].flatten().tolist() if isinstance(smpl_data["Rh"], np.ndarray) else smpl_data["Rh"]
        betas = smpl_data["betas"].flatten().tolist() if isinstance(smpl_data["betas"], np.ndarray) else smpl_data["betas"]
        
        response_data = {
            "frameCount": result["frame_count"],
            "fps": 30,
            "duration": result["output_duration"],
            "poses": poses,
            "trans": trans,
            "Rh": Rh,
            "betas": betas,
            "gender": smpl_data.get("gender", "neutral"),
            "text": request.prompt,
            "seed": result["seed"],
            "cfg_scale": request.cfg_scale,
            "generation_time": result["generation_time"],
            "timestamp": timestamp,
            "file_path": base_filename,
        }
        
        return GenerateResponse(
            success=True,
            message=f"Generated {result['frame_count']} frames ({result['output_duration']}s) in {result['generation_time']:.2f}s",
            data=response_data,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate")
async def generate_motion_get(
    prompt: str = Query(..., description="Text prompt describing the motion"),
    duration: float = Query(default=5.0, ge=1.0, le=10.0, description="Motion duration in seconds"),
    seed: int = Query(default=0, ge=0, description="Random seed (0 for random)"),
    cfg_scale: float = Query(default=7.0, ge=1.0, le=10.0, description="Classifier-free guidance scale"),
):
    """
    Generate motion from a text prompt (GET version for easy testing).
    """
    request = GenerateRequest(
        prompt=prompt,
        duration=duration,
        seed=seed,
        cfg_scale=cfg_scale,
    )
    return await generate_motion(request)


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("HY-Motion Streaming Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /              - Health check")
    print("  GET  /health        - Detailed health check")
    print("  POST /generate      - Generate motion (JSON body)")
    print("  GET  /generate      - Generate motion (query params)")
    print("\nExample:")
    print('  curl "http://localhost:7860/generate?prompt=a%20person%20walking"')
    print('  curl -X POST http://localhost:7860/generate -H "Content-Type: application/json" \\')
    print('       -d \'{"prompt": "a person dancing", "duration": 5, "cfg_scale": 7}\'')
    print("=" * 60)
    
    uvicorn.run(
        "streaming_server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
