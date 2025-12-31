#!/usr/bin/env python3
"""
Streaming Gradio Interface for HY-Motion-1.0

This interface provides real-time frame-by-frame streaming of motion generation,
allowing users to see the motion being generated progressively.
"""

import argparse
import codecs as cs
import json
import os
import os.path as osp
import random
import re
import textwrap
import time
from typing import List, Optional, Tuple, Union, Generator, Dict, Any
import threading
import queue

import torch
import numpy as np
from huggingface_hub import snapshot_download

import gradio as gr

# Import the runtime and necessary utilities
from hymotion.utils.t2m_runtime import T2MRuntime
from hymotion.utils.visualize_mesh_web import generate_static_html_content
from hymotion.pipeline.motion_diffusion import MotionFlowMatching
from hymotion.utils.type_converter import get_module_device
from hymotion.pipeline.motion_diffusion import length_to_mask

# Global runtime instance
_runtime_instance: Optional[T2MRuntime] = None
_runtime_lock = threading.Lock()

# Frame queue for streaming
_frame_queue = queue.Queue()
_stop_streaming = threading.Event()

def _init_runtime_if_needed() -> 'StreamingT2MRuntime':
    """Initialize the runtime instance if not already created."""
    global _runtime_instance
    
    with _runtime_lock:
        if _runtime_instance is None:
            print(">>> Initializing StreamingT2MRuntime...")
            # Use default configuration similar to local_infer.py
            
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
                # Use default path and let it try to download
                found_path = "ckpts/tencent/HY-Motion-1.0"
                print(f">>> [WARNING] Model files not found in any expected location")
                print(">>> [WARNING] Will attempt to load from Hugging Face")
            
            cfg = os.path.join(found_path, "config.yml")
            ckpt = os.path.join(found_path, "latest.ckpt")
            
            # Check if we should skip model loading
            skip_model_loading = not (os.path.exists(cfg) and os.path.exists(ckpt))
            
            # Initialize with default settings - force GPU 0
            print(">>> [INFO] Configuring model to load on GPU 0")
            _runtime_instance = StreamingT2MRuntime(
                config_path=cfg,
                ckpt_name=ckpt,
                device_ids=[0],  # Force GPU 0
                disable_prompt_engineering=True,  # Disable for streaming demo
                skip_model_loading=skip_model_loading,
            )
    
    return _runtime_instance


class StreamingT2MRuntime(T2MRuntime):
    """
    Extended T2MRuntime that supports frame-by-frame streaming generation.
    """

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
        # Note: Quantization disabled for now due to dtype mismatches in ODE solver
        # Diffusion models require careful mixed precision handling
        print(">>> [INFO] Loading model in FP32 on GPU 0 (full precision)")
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
    
    def generate_motion_streaming(
        self,
        text: str,
        seeds_csv: str,
        duration: float,
        cfg_scale: float,
        output_format: str = "dict",
        use_special_game_feat: bool = False,
        frame_callback=None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate motion frame by frame with streaming support.
        
        Args:
            text: Input text prompt
            seeds_csv: Comma-separated seed values
            duration: Motion duration in seconds
            cfg_scale: Classifier-free guidance scale
            output_format: Output format (dict, fbx, etc.)
            use_special_game_feat: Use special game features
            frame_callback: Optional callback for each generated frame
            
        Yields:
            Dictionary containing frame data and metadata
        """
        self.load()
        seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip() != ""]
        pi = self._acquire_pipeline()
        
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()
            
            # Get the pipeline parameters
            device = get_module_device(pipeline)
            length = int(round(duration * pipeline.output_mesh_fps))
            
            # Ensure length is within bounds
            if length > pipeline.train_frames or length < min(pipeline.train_frames, 20):
                print(f">>> given length is too long or too short, got {length}, will be truncated")
                length = min(length, pipeline.train_frames)
                length = max(length, min(pipeline.train_frames, 20))
            
            # Set up text encoding
            text_list = [text] * len(seeds)
            hidden_state_dict = pipeline.encode_text({"text": text_list})
            
            # Prepare inputs
            vtxt_input = hidden_state_dict["text_vec_raw"]
            ctxt_input = hidden_state_dict["text_ctxt_raw"]
            ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
            
            # Set up masks
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            x_length = torch.LongTensor([length] * len(seeds)).to(device)
            x_mask_temporal = length_to_mask(x_length, pipeline.train_frames)
            
            # Set up classifier-free guidance
            text_guidance_scale = cfg_scale if cfg_scale is not None else pipeline.text_guidance_scale
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
            
            # Use the proper ODE solver like non-streaming version
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

            # Generate the full trajectory using proper ODE solver
            print(f">>> Starting ODE solver for {length} frames ({duration}s @ {pipeline.output_mesh_fps}fps)...")
            ode_start_time = time.time()

            with torch.no_grad():
                trajectory = odeint(fn, y0, t, **pipeline._noise_scheduler_cfg)

            ode_end_time = time.time()
            ode_duration = ode_end_time - ode_start_time
            print(f">>> ODE solver completed in {ode_duration:.2f} seconds")
            print(f">>> Generated {length} frames ({duration}s output) in {ode_duration:.2f}s (speed: {duration/ode_duration:.2f}x realtime)")

            # Now stream the results
            with torch.no_grad():
                # Stream each time step from the trajectory
                for i in range(1, len(trajectory)):  # Skip initial noise
                    current_x = trajectory[i]

                    # Yield intermediate results
                    intermediate_sampled = current_x[:, :length, ...].clone()
                    intermediate_output = pipeline.decode_motion_from_latent(
                        intermediate_sampled,
                        should_apply_smooothing=False  # Skip smoothing for intermediate frames
                    )

                    frame_data = {
                        "frame_index": i,
                        "total_frames": len(t) - 1,
                        "progress": i / (len(t) - 1),
                        "motion_data": intermediate_output,
                        "text": text,
                        "timestamp": time.time()
                    }

                    # Print SMPL-H data for this frame
                    print(f"\n=== Frame {i}/{len(t)-1} SMPL-H Data ===")
                    print(f"Progress: {i/(len(t)-1):.1%}")

                    # Extract and print key motion data
                    if hasattr(intermediate_output, 'keys'):
                        k3d = intermediate_output.get('keypoints3d', None)
                        transl = intermediate_output.get('transl', None)
                        rot6d = intermediate_output.get('rot6d', None)

                        if k3d is not None:
                            print(f"Keypoints3D shape: {k3d.shape} (Batch, Frames, Joints, XYZ)")
                            print(f"First frame keypoints sample: {k3d[0, 0, :3, :]} (first 3 joints)")

                        if transl is not None:
                            print(f"Translation shape: {transl.shape} (Batch, Frames, XYZ)")
                            print(f"First frame translation: {transl[0, 0, :]}")

                        if rot6d is not None:
                            print(f"Rotation6D shape: {rot6d.shape} (Batch, Frames, Joints, 6D)")
                            print(f"First frame root rotation: {rot6d[0, 0, 0, :]}")

                    yield frame_data

                    # Check if we should stop streaming
                    if _stop_streaming.is_set():
                        break
                
                # Final result with smoothing
                if not _stop_streaming.is_set():
                    sampled = trajectory[-1][:, :length, ...].clone()
                    final_output = pipeline.decode_motion_from_latent(
                        sampled,
                        should_apply_smooothing=True
                    )
                    
                    final_data = {
                        "frame_index": len(t) - 1,
                        "total_frames": len(t) - 1,
                        "progress": 1.0,
                        "motion_data": final_output,
                        "text": text,
                        "timestamp": time.time(),
                        "completed": True,
                        "generation_time": ode_duration,
                        "output_duration": duration,
                    }
                    
                    # Print final SMPL-H data and timing summary
                    print(f"\n=== Final Frame SMPL-H Data ===")
                    print(f"Generation completed!")
                    print(f"\n=== Generation Summary ===")
                    print(f"Output Duration: {duration}s ({length} frames @ {pipeline.output_mesh_fps}fps)")
                    print(f"Generation Time: {ode_duration:.2f}s")
                    print(f"Speed: {duration/ode_duration:.2f}x realtime")
                    print(f"Text Prompt: {text}")
                    
                    if hasattr(final_output, 'keys'):
                        k3d = final_output.get('keypoints3d', None)
                        transl = final_output.get('transl', None)
                        rot6d = final_output.get('rot6d', None)
                        
                        if k3d is not None:
                            print(f"Final Keypoints3D shape: {k3d.shape}")
                            print(f"Final frame keypoints sample: {k3d[0, -1, :3, :]} (last frame, first 3 joints)")
                        
                        if transl is not None:
                            print(f"Final Translation shape: {transl.shape}")
                            print(f"Final frame translation: {transl[0, -1, :]}")
                        
                        if rot6d is not None:
                            print(f"Final Rotation6D shape: {rot6d.shape}")
                            print(f"Final frame root rotation: {rot6d[0, -1, 0, :]}")
                    
                    yield final_data
                    
        finally:
            self._release_pipeline(pi)




def streaming_generate_motion(
    text: str,
    seeds_csv: str,
    motion_duration: float,
    cfg_scale: float,
    output_format: str,
    original_text: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streaming version of motion generation that yields HTML updates.
    
    Args:
        text: Input text prompt
        seeds_csv: Comma-separated seed values
        motion_duration: Duration in seconds
        cfg_scale: CFG scale
        output_format: Output format
        original_text: Original text (before rewriting)
        output_dir: Output directory
        
    Yields:
        HTML content updates for Gradio streaming
    """
    runtime = _init_runtime_if_needed()
    
    # Clear any previous frames
    while not _frame_queue.empty():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            break
    
    _stop_streaming.clear()
    
    try:
        # Start the streaming generation in a separate thread
        def generation_thread():
            try:
                frame_generator = runtime.generate_motion_streaming(
                    text=text,
                    seeds_csv=seeds_csv,
                    duration=motion_duration,
                    cfg_scale=cfg_scale,
                    output_format=output_format,
                )
                
                for frame_data in frame_generator:
                    if _stop_streaming.is_set():
                        print(">>> Generation stopped by user")
                        break

                    frame_idx = frame_data['frame_index']
                    total_frames = frame_data['total_frames']
                    progress = frame_data['progress']

                    print(f">>> Frame {frame_idx}/{total_frames} generated (progress: {progress:.1%})")

                    # Create a simple HTML representation of the current frame
                    frame_html = f"""
                    <div style="text-align: center; padding: 20px;">
                        <h3>🎬 Streaming Motion Generation</h3>
                        <div style="margin: 20px 0;">
                            <strong>Prompt:</strong> {text}<br>
                            <strong>Progress:</strong> {progress:.1%}<br>
                            <strong>Frame:</strong> {frame_idx}/{total_frames}
                        </div>
                        <div style="background: #f0f0f0; padding: 20px; border-radius: 10px;">
                            <p>Generating motion frame by frame...</p>
                            <div style="background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;">
                                <div style="background: #4CAF50; height: 100%; width: {progress*100}%; transition: width 0.3s;"></div>
                            </div>
                        </div>
                        <div style="margin-top: 20px; font-size: 14px; color: #666;">
                            <p>Streaming mode: Watch as your motion is generated in real-time!</p>
                            <p style="font-family: monospace; font-size: 12px; color: #999;">
                                Timestamp: {frame_data.get('timestamp', time.time())}
                            </p>
                        </div>
                    </div>
                    """
                    print(f">>> Putting frame HTML in queue (size: {_frame_queue.qsize()})")
                    _frame_queue.put(frame_html)
                    
                    if frame_data.get("completed"):
                        # Final result - generate the full visualization
                        try:
                            from hymotion.utils.visualize_mesh_web import save_visualization_data

                            motion_data = frame_data["motion_data"]
                            timestamp = str(int(time.time()))
                            file_path = f"streaming_{timestamp}"

                            # Debug: Check what's in motion_data
                            print(f">>> Motion data keys: {motion_data.keys() if hasattr(motion_data, 'keys') else 'not a dict'}")
                            if hasattr(motion_data, 'keys'):
                                for key in motion_data.keys():
                                    val = motion_data[key]
                                    if hasattr(val, 'shape'):
                                        print(f">>>   {key}: shape={val.shape}")
                                    else:
                                        print(f">>>   {key}: type={type(val)}")

                            # Save the motion data to disk first (let it use default output_dir)
                            print(f">>> Saving visualization data...")
                            memory_data, base_filename = save_visualization_data(
                                output=motion_data,
                                text=text,
                                rewritten_text=text,  # Use same text since prompt engineering is disabled
                                timestamp=timestamp,
                                output_dir=None,  # Use default
                                output_filename=file_path,
                            )
                            print(f">>> Data saved with base filename: {base_filename}")

                            # Now generate HTML from the saved data (let it use default output_dir)
                            print(f">>> Generating HTML visualization...")
                            html_content = runtime._generate_html_content(
                                timestamp=timestamp,
                                file_path=base_filename,
                                output_dir=None,  # Use default
                            )
                            print(f">>> HTML generation successful, length: {len(html_content)}")
                            
                            # Use proper HTML escaping for srcdoc
                            import html as html_module
                            escaped_html = html_module.escape(html_content, quote=True)

                            iframe_html = f"""
                                <iframe
                                    srcdoc="{escaped_html}"
                                    width="100%"
                                    height="750px"
                                    style="border: none; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);"
                                    sandbox="allow-scripts allow-same-origin"
                                ></iframe>
                            """
                            print(f">>> Putting iframe HTML in queue (escaped size: {len(iframe_html)} bytes)")
                            _frame_queue.put(iframe_html)
                        except Exception as e:
                            print(f">>> HTML generation failed: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fallback: Create a simple completion message
                            gen_time = frame_data.get('generation_time', 0)
                            out_dur = frame_data.get('output_duration', motion_duration)
                            speed_ratio = out_dur / gen_time if gen_time > 0 else 0
                            fallback_html = f"""
                            <div style='text-align: center; padding: 40px; background: #e8f5e9; border-radius: 12px;'>
                                <h3 style='color: #2e7d32;'>✅ Motion Generation Completed!</h3>
                                <p style='font-size: 16px; color: #1b5e20; margin: 20px 0;'>
                                    Your motion has been generated successfully.<br>
                                    Output Duration: {out_dur:.1f}s | CFG Scale: {cfg_scale}
                                </p>
                                <div style='background: #c8e6c9; padding: 15px; border-radius: 8px; margin: 20px 0;'>
                                    <strong>Prompt:</strong> {text}
                                </div>
                                <div style='background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; color: #856404;'>
                                    <strong>⏱️ Generation Time:</strong> {gen_time:.2f}s<br>
                                    <strong>⚡ Speed:</strong> {speed_ratio:.2f}x realtime
                                </div>
                                <div style='margin: 20px 0;'>
                                    <button onclick='window.location.reload()' style='
                                        background: #4caf50; 
                                        color: white; 
                                        border: none; 
                                        padding: 12px 24px; 
                                        border-radius: 6px; 
                                        cursor: pointer; 
                                        font-size: 16px; 
                                        transition: background 0.3s;
                                    ' onmouseover='this.style.background="#388e3c"' onmouseout='this.style.background="#4caf50"'>
                                        🔄 Generate New Motion
                                    </button>
                                </div>
                            </div>
                            """
                            _frame_queue.put(fallback_html)
                        break
                        
            except Exception as e:
                error_html = f"""
                <div style="color: red; padding: 20px; text-align: center;">
                    <h3>❌ Error during streaming generation</h3>
                    <p>{str(e)}</p>
                </div>
                """
                _frame_queue.put(error_html)
        
        # Start the generation thread
        thread = threading.Thread(target=generation_thread, daemon=True)
        thread.start()
        
        # Initial waiting message
        yield f"""
        <div style="text-align: center; padding: 40px;">
            <h3>🕒 Initializing motion generation...</h3>
            <p>Please wait while we set up the generation pipeline...</p>
            <div style="margin: 20px;">
                <div style="font-family: monospace; font-size: 18px;">⏳</div>
            </div>
        </div>
        """

        # Yield updates from the queue
        last_update = None
        update_count = 0
        while True:
            if _stop_streaming.is_set():
                print(">>> Streaming stopped")
                yield "<div style='color: orange; text-align: center; padding: 20px;'>Generation stopped by user.</div>"
                break

            try:
                # Try to get an update with a timeout
                html_update = _frame_queue.get(timeout=0.5)
                last_update = html_update
                update_count += 1
                print(f">>> Yielding update #{update_count} to Gradio (length: {len(html_update)} chars)")
                yield html_update

                # If this was the final result, we're done
                if "</iframe>" in html_update or "Generation Completed" in html_update:
                    print(f">>> Final update detected, stopping stream")
                    break

            except queue.Empty:
                # No update yet, check if thread is still alive
                if not thread.is_alive():
                    print(f">>> Thread completed, yielded {update_count} updates total")
                    if last_update is not None:
                        break
                    else:
                        # Thread died without producing output
                        yield "<div style='color: red; text-align: center; padding: 20px;'>Generation failed - thread ended without output</div>"
                        break
                # Continue waiting for updates
                continue
            except Exception as e:
                print(f">>> Error in streaming loop: {e}")
                yield f"<div style='color: red; text-align: center; padding: 20px;'>Error: {str(e)}</div>"
                break

        # Wait for thread to complete
        print(f">>> Waiting for thread to join...")
        thread.join(timeout=5)
        print(f">>> Thread joined, streaming complete")
                
    finally:
        _stop_streaming.set()


class StreamingGradioApp:
    """Streaming Gradio interface for HY-Motion-1.0."""
    
    def __init__(self):
        self.runtime = None
        self.prompt_engineering_available = False
        self.fbx_available = False
        
        # Initialize runtime
        try:
            self.runtime = _init_runtime_if_needed()
            self.fbx_available = getattr(self.runtime, "fbx_available", False)
            
            # Check if prompt engineering is available
            try:
                if hasattr(self.runtime, 'prompt_rewriter') and self.runtime.prompt_rewriter is not None:
                    self.prompt_engineering_available = True
            except:
                self.prompt_engineering_available = False
                
        except Exception as e:
            print(f">>> Failed to initialize runtime: {e}")
            import traceback
            traceback.print_exc()
            self.runtime = None
    
    def build_interface(self):
        """Build the Gradio interface."""
        
        with gr.Blocks(css=self._get_css()) as demo:
            
            # Header
            gr.Markdown(self._get_header_md(), elem_classes=["main-header"])
            
            with gr.Row():
                # Left control panel
                with gr.Column(scale=2, elem_classes=["left-panel"]):
                    
                    # Input controls
                    self.text_input = gr.Textbox(
                        label="📝 Input Text",
                        placeholder="Enter text to generate motion, support Chinese and English text input.",
                    )
                    
                    self.duration_slider = gr.Slider(
                        minimum=0.5,
                        maximum=12,
                        value=5.0,
                        step=0.1,
                        label="⏱️ Action Duration (seconds)",
                        info="Feel free to adjust the action duration",
                    )
                    
                    self.seed_input = gr.Textbox(
                        label="🎲 Random Seeds",
                        placeholder="Comma-separated seeds (e.g., 42, 123, 456)",
                        value="42",
                    )
                    
                    self.cfg_slider = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=7.0,
                        step=0.5,
                        label="🎨 CFG Scale",
                        info="Higher values follow the prompt more closely",
                    )
                    
                    # Generation buttons
                    with gr.Row():
                        self.stop_btn = gr.Button(
                            "⏹️ Stop Generation",
                            variant="stop",
                            size="lg",
                        )
                        
                        self.generate_btn = gr.Button(
                            "🚀 Generate Motion (Streaming)",
                            variant="primary",
                            size="lg",
                        )
                    
                    # Status display
                    self.status_output = gr.Textbox(
                        label="📊 Status Information",
                        value="Enter your text and click [🚀 Generate Motion (Streaming)] to start frame-by-frame generation.",
                    )
                    
                    # Advanced settings
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        self._build_advanced_settings()
                    
                # Right display area
                with gr.Column(scale=3):
                    self.output_display = gr.HTML(
                        value=self._get_placeholder_html(),
                        show_label=False,
                        elem_classes=["flask-display"]
                    )
            
            # Set up event handlers
            self._setup_event_handlers()
            
        return demo
        
    def _get_css(self) -> str:
        """Get CSS styles for the interface."""
        return """
        /* Streaming-specific styles */
        .streaming-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .streaming-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        
        /* Progress bar animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        """
    
    def _get_header_md(self) -> str:
        """Get header markdown."""
        return """
        # 🎬 HY-Motion 1.0 - Streaming Interface
        
        **Real-time Frame-by-Frame Motion Generation**
        
        Watch your motion being generated progressively, frame by frame!
        """
    
    def _get_placeholder_html(self) -> str:
        """Get placeholder HTML for the output display."""
        return """
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #667eea; margin-bottom: 20px;">🎬 Streaming Motion Generation</h2>
            <p style="font-size: 16px; color: #666; margin-bottom: 30px;">
                This interface generates motion frame by frame, allowing you to see the progression in real-time.
            </p>
            <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; display: inline-block;">
                <div style="font-size: 48px; margin-bottom: 15px;">🕒</div>
                <p style="margin: 0; color: #888;">Waiting for your prompt...</p>
            </div>
            <div style="margin-top: 30px; padding: 20px; background: #e8f4f8; border-radius: 8px; display: inline-block;">
                <strong>How it works:</strong><br>
                1. Enter your text prompt<br>
                2. Click "Generate Motion (Streaming)"<br>
                3. Watch frames appear progressively!<br>
            </div>
        </div>
        """
    
    def _build_advanced_settings(self):
        """Build advanced settings section."""
        # Add any streaming-specific advanced settings here
        pass
        
    def _setup_event_handlers(self):
        """Set up Gradio event handlers."""
        
        # Check if runtime is available
        if self.runtime is None:
            print(">>> WARNING: Runtime not initialized. Streaming generation will fail.")
            # Show error message to user
            self.status_output.value = "❌ Runtime not initialized. Please check console for errors."
        
        # Stop button handler
        self.stop_btn.click(
            fn=lambda: _stop_streaming.set(),
            outputs=[self.status_output]
        ).then(
            fn=lambda: "⏹️ Generation stopped by user.",
            outputs=[self.status_output]
        )
        
        # Generate button handler
        self.generate_btn.click(
            fn=lambda: "🚀 Starting streaming generation...",
            outputs=[self.status_output]
        ).then(
            fn=streaming_generate_motion,
            inputs=[
                self.text_input,
                self.seed_input,
                self.duration_slider,
                self.cfg_slider,
                gr.State("dict")  # output_format
            ],
            outputs=[self.output_display]
        ).then(
            fn=lambda: "🎉 Streaming generation completed!",
            outputs=[self.status_output]
        )


def main(args):
    """Main function to run the streaming Gradio app."""
    
    # Initialize the streaming app
    app = StreamingGradioApp()
    demo = app.build_interface()
    
    # Try to find an available port
    import socket
    
    def find_available_port(start_port):
        """Find an available port starting from start_port."""
        port = start_port
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                port += 1
                if port > start_port + 10:  # Try up to 10 ports
                    return None
    
    # Use the specified port or find an available one
    target_port = args.port
    if not args.port:
        target_port = 7860
    
    available_port = find_available_port(target_port)
    if available_port and available_port != target_port:
        print(f">>> Port {target_port} is in use, trying port {available_port}")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=available_port or target_port,  # Use available port or original
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="HY-Motion 1.0 Streaming Gradio Interface")
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the Gradio interface on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    args = parser.parse_args()
    
    print(f">>> Starting HY-Motion 1.0 Streaming Interface on port {args.port}")
    main(args)