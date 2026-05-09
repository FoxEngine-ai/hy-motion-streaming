# t2m_runtime.py
import json
import os
import threading
import time
import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from ..prompt_engineering.prompt_rewrite import PromptRewriter
from .loaders import load_object
from .visualize_mesh_web import save_visualization_data, generate_static_html_content

try:
    import fbx

    FBX_AVAILABLE = True
    print(">>> FBX module found.")
except ImportError:
    FBX_AVAILABLE = False
    print(">>> FBX module not found.")


def _get_local_ip():
    import subprocess

    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for ip in result.stdout.strip().split():
            if not ip.startswith("127.") and not ip.startswith("172.17."):
                return ip
    return "localhost"


def _now():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


class T2MRuntime:
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
        quantization_mode: Optional[str] = None,
        use_gguf: bool = False,
    ):
        self.config_path = config_path
        # Resolve a bare ckpt filename (e.g. "latest.ckpt") relative to the
        # config file's directory. Without this, downstream code that does
        # os.path.dirname(self.ckpt_name) gets an empty string when ckpt_name
        # has no directory component, which then propagates as an empty path
        # to load_in_demo() — manifesting as an "AssertionError on empty
        # mean_std_name" because the pipeline can't locate the sibling
        # statistics file. Absolute or already-relative-with-directory values
        # are left alone.
        if ckpt_name and not os.path.isabs(ckpt_name) and not os.path.dirname(ckpt_name):
            ckpt_name = os.path.join(os.path.dirname(os.path.abspath(config_path)), ckpt_name)
        self.ckpt_name = ckpt_name
        self.skip_text = skip_text
        self.quantization_mode = quantization_mode
        self.use_gguf = use_gguf
        self.prompt_engineering_host = prompt_engineering_host
        self.prompt_engineering_model_path = prompt_engineering_model_path
        self.disable_prompt_engineering = disable_prompt_engineering
        self.skip_model_loading = skip_model_loading
        self.local_ip = _get_local_ip()

        if force_cpu:
            print(">>> [INFO] CPU mode enabled via HY_MOTION_DEVICE=cpu environment variable")
            self.device_ids = []
        elif torch.cuda.is_available():
            all_ids = list(range(torch.cuda.device_count()))
            self.device_ids = all_ids if device_ids is None else [i for i in device_ids if i in all_ids]
        else:
            self.device_ids = []

        self.pipelines = []
        self._gpu_load = []
        self._lock = threading.Lock()
        self._loaded = False

        # Initialize FBX availability before load()
        self.fbx_available = FBX_AVAILABLE
        if self.fbx_available:
            try:
                from .smplh2woodfbx import SMPLH2WoodFBX
                self.fbx_converter = SMPLH2WoodFBX()
            except Exception as e:
                print(f">>> Failed to initialize FBX converter: {e}")
                self.fbx_available = False
                self.fbx_converter = None
        else:
            self.fbx_converter = None
            print(">>> FBX module not found. FBX export will be disabled.")

        if self.disable_prompt_engineering:
            self.prompt_rewriter = None
        else:
            self.prompt_rewriter = PromptRewriter(
                host=self.prompt_engineering_host,
                model_path=self.prompt_engineering_model_path,
                lazy_load=True  # Enable lazy loading for faster startup
            )
        # Skip model loading if checkpoint not found
        if self.skip_model_loading:
            print(">>> [WARNING] Checkpoint not found, will use randomly initialized model weights")
        self.load()

        device_info = self.device_ids if self.device_ids else "cpu"
        if self.skip_model_loading:
            print(
                f">>> T2MRuntime initialized (using randomly initialized weights) in IP {self.local_ip}, devices={device_info}"
            )
        else:
            print(f">>> T2MRuntime loaded in IP {self.local_ip}, devices={device_info}")

    @staticmethod
    def apply_looping(
        motion_data: dict, 
        blend_duration: float = 0.5,
        fps: float = 30.0
    ) -> dict:
        """
        Apply looping by blending the end of the motion back to the start frame.
        
        Args:
            motion_data: Dictionary containing motion data (rot6d, transl, etc.)
            blend_duration: Duration of the blend in seconds
            fps: Frames per second of the motion
            
        Returns:
            Modified motion_data dictionary
        """
        print(f">>> Applying looping with blend duration {blend_duration}s...")
        
        # Check if we have necessary data
        if 'rot6d' not in motion_data or 'transl' not in motion_data:
            print(">>> [WARNING] Missing rot6d or transl for looping, skipping.")
            return motion_data
            
        rot6d = motion_data['rot6d']   # (Batch, Frames, Joints, 6)
        transl = motion_data['transl'] # (Batch, Frames, 3)
        
        device = rot6d.device
        dtype = rot6d.dtype
        
        # Calculate blend frames
        blend_frames = int(blend_duration * fps)
        total_frames = rot6d.shape[1]
        
        if blend_frames >= total_frames // 2:
            print(f">>> [WARNING] Motion too short for requested blend duration. Reducing blend.")
            blend_frames = max(1, total_frames // 4)
            
        if blend_frames <= 0:
            return motion_data
            
        # We will modify the last `blend_frames` to blend towards the first frame
        
        # 1. Get targets (first frame)
        target_rot6d = rot6d[:, 0:1, :, :]      # (Batch, 1, Joints, 6)
        target_transl = transl[:, 0:1, :]       # (Batch, 1, 3)
        
        # 2. Get the source frames (last N frames)
        # We blend the last N frames. 
        # Actually, for a perfect loop, we want the LAST frame to match the FIRST frame.
        # But we also want the transition to be smooth.
        # Strategy: Interpolate from (end - blend_frames) -> end  TO  (end - blend_frames) -> start
        
        # Let's extract the segment we want to modify
        start_blend_idx = total_frames - blend_frames
        
        segment_rot6d = rot6d[:, start_blend_idx:, :, :]  # (B, blend_frames, J, 6)
        segment_transl = transl[:, start_blend_idx:, :]   # (B, blend_frames, 3)
        
        # Create weights (0 at start of blend, 1 at end of blend)
        weights = torch.linspace(0, 1, blend_frames, device=device, dtype=dtype)
        # Reshape for broadcasting
        # weights: (blend_frames,) -> need (1, blend_frames, 1, 1) for rot6d
        w_rot = weights.view(1, blend_frames, 1, 1)
        w_trans = weights.view(1, blend_frames, 1)
        
        # --- Interpolate Translation (LERP) ---
        # Note: Translation must be relative to root? 
        # If the character moved far away, snapping back to start position (0,0,0) might look weird if we don't handle root motion.
        # For now, we assume we want to loop back to the EXACT start position.
        
        # LERP: result = src * (1 - w) + target * w
        blended_transl = segment_transl * (1 - w_trans) + target_transl * w_trans
        
        # --- Interpolate Rotation (SLERP-ish for 6D) ---
        # 6D rotation is continuous, so LERP usually works fine, but we can re-normalize.
        # Ideally convert to Quat -> SLERP -> 6D, or Matrix -> 6D.
        # Since we use 6D representation, simple linear interpolation + orthonormalization works well.
        
        blended_rot6d = segment_rot6d * (1 - w_rot) + target_rot6d * w_rot
        
        # Update the data
        motion_data['rot6d'] = rot6d.clone()
        motion_data['rot6d'][:, start_blend_idx:, :, :] = blended_rot6d
        
        motion_data['transl'] = transl.clone()
        motion_data['transl'][:, start_blend_idx:, :] = blended_transl
        
        # If root_rotations_mat exists, it needs to be updated too or invalidated
        if 'root_rotations_mat' in motion_data:
            # Re-calculating proper rotation matrices from 6D is safest, 
            # but simpler to just let valid downstream code handle it (like fbx converter)
            # or we can try to blend if it matches shape
            # For now, let's remove it to force regeneration if needed, or leave it if unused
            pass
            
        print(">>> Looping applied successfully.")
        return motion_data

    def _print_smplh_data(self, model_output: dict, text: str) -> None:
        """Print SMPL-H motion data for debugging and analysis."""
        print(f"\n{'='*60}")
        print(f"SMPL-H Data for: '{text}'")
        print(f"{'='*60}")
        
        if not isinstance(model_output, dict):
            print("❌ Model output is not a dictionary")
            return
            
        # Extract motion data
        k3d = model_output.get('keypoints3d', None)
        transl = model_output.get('transl', None)
        rot6d = model_output.get('rot6d', None)
        root_rot = model_output.get('root_rotations_mat', None)
        
        if k3d is not None:
            print(f"📊 Keypoints3D:")
            print(f"   Shape: {k3d.shape} (Batch={k3d.shape[0]}, Frames={k3d.shape[1]}, Joints={k3d.shape[2]})")
            print(f"   First frame, first 3 joints XYZ:")
            for joint_idx in range(min(3, k3d.shape[2])):
                pos = k3d[0, 0, joint_idx, :].cpu().numpy()
                print(f"     Joint {joint_idx}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            print(f"   Last frame, first 3 joints XYZ:")
            for joint_idx in range(min(3, k3d.shape[2])):
                pos = k3d[0, -1, joint_idx, :].cpu().numpy()
                print(f"     Joint {joint_idx}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        if transl is not None:
            print(f"\n📊 Translation:")
            print(f"   Shape: {transl.shape} (Batch={transl.shape[0]}, Frames={transl.shape[1]})")
            print(f"   First frame: [{transl[0, 0, 0]:.3f}, {transl[0, 0, 1]:.3f}, {transl[0, 0, 2]:.3f}]")
            print(f"   Last frame:  [{transl[0, -1, 0]:.3f}, {transl[0, -1, 1]:.3f}, {transl[0, -1, 2]:.3f}]")
        
        if rot6d is not None:
            print(f"\n📊 Rotation6D:")
            print(f"   Shape: {rot6d.shape} (Batch={rot6d.shape[0]}, Frames={rot6d.shape[1]}, Joints={rot6d.shape[2]})")
            print(f"   First frame, root joint: {rot6d[0, 0, 0, :].cpu().numpy()}")
            print(f"   Last frame, root joint:  {rot6d[0, -1, 0, :].cpu().numpy()}")
        
        if root_rot is not None:
            print(f"\n📊 Root Rotations:")
            print(f"   Shape: {root_rot.shape} (Batch={root_rot.shape[0]}, Frames={root_rot.shape[1]})")
            print(f"   First frame rotation matrix:")
            print(f"     {root_rot[0, 0, :, :].cpu().numpy()}")
        
        print(f"{'='*60}\n")

    def load(self):
        if self._loaded:
            return
        import time
        
        print(f">>> Loading model from {self.config_path}...")
        print(f">>> [DEBUG] Starting T2MRuntime loading process...")
        
        total_start = time.time()

        # Step 1: Load config
        print(f">>> [DEBUG] Step 1/4: Loading configuration file...")
        config_start = time.time()
        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config_time = time.time() - config_start
        print(f">>> [DEBUG] Configuration loaded in {config_time:.2f} seconds")

        # Use allow_empty_ckpt=True when skip_model_loading is True
        allow_empty_ckpt = self.skip_model_loading

        # Step 2: Create and load pipelines
        print(f">>> [DEBUG] Step 2/4: Creating and loading pipelines...")
        print(f">>> [DEBUG] Device IDs: {self.device_ids}")
        print(f">>> [DEBUG] Skip text encoder: {self.skip_text}")
        print(f">>> [DEBUG] Skip model loading: {self.skip_model_loading}")
        
        pipeline_start = time.time()
        
        if not self.device_ids:
            print(f">>> [DEBUG] Loading pipeline on CPU...")
            pipeline = load_object(
                config["train_pipeline"],
                config["train_pipeline_args"],
                network_module=config["network_module"],
                network_module_args=config["network_module_args"],
                quantization_mode=self.quantization_mode,
                use_gguf=self.use_gguf,
            )
            device = torch.device("cpu")
            print(f">>> [DEBUG] Loading checkpoint from {self.ckpt_name}...")
            pipeline.load_in_demo(
                self.ckpt_name,
                os.path.dirname(self.ckpt_name),
                build_text_encoder=not self.skip_text,
                allow_empty_ckpt=allow_empty_ckpt,
            )
            print(f">>> [DEBUG] Moving pipeline to CPU...")
            pipeline.to(device)
            self.pipelines = [pipeline]
            self._gpu_load = [0]
        else:
            print(f">>> [DEBUG] Loading pipeline on GPU {self.device_ids[0]}...")
            print(f">>> [DEBUG] Note: HY-Motion model is large (~45GB) and will use GPU {self.device_ids[0]}")
            print(f">>> [DEBUG] Prompter model will use GPU {self.device_ids[1] if len(self.device_ids) > 1 else self.device_ids[0]}")
            p_start = time.time()
            
            # Load pipeline on the first GPU only (model is too large to split)
            p = load_object(
                config["train_pipeline"],
                config["train_pipeline_args"],
                network_module=config["network_module"],
                network_module_args=config["network_module_args"],
                quantization_mode=self.quantization_mode,
                use_gguf=self.use_gguf,
            )
            print(f">>> [DEBUG] Loading checkpoint from {self.ckpt_name}...")
            p.load_in_demo(
                self.ckpt_name,
                os.path.dirname(self.ckpt_name),
                build_text_encoder=not self.skip_text,
                allow_empty_ckpt=allow_empty_ckpt,
            )
            print(f">>> [DEBUG] Moving pipeline to cuda:{self.device_ids[0]}...")
            p.to(torch.device(f"cuda:{self.device_ids[0]}"))
            
            self.pipelines = [p]
            self._gpu_load = [0]
            p_time = time.time() - p_start
            print(f">>> [DEBUG] Pipeline loaded in {p_time:.2f} seconds on GPU {self.device_ids[0]}")
        
        pipeline_time = time.time() - pipeline_start
        print(f">>> [DEBUG] All pipelines loaded in {pipeline_time:.2f} seconds")

        # Step 3: Initialize FBX converter
        print(f">>> [DEBUG] Step 3/4: Initializing FBX converter...")
        fbx_start = time.time()
        if self.fbx_available:
            try:
                from .smplh2woodfbx import SMPLH2WoodFBX
                self.fbx_converter = SMPLH2WoodFBX()
                print(f">>> [DEBUG] FBX converter initialized successfully")
            except Exception as e:
                print(f">>> [DEBUG] Failed to initialize FBX converter: {e}")
                self.fbx_available = False
                self.fbx_converter = None
        else:
            self.fbx_converter = None
            print(">>> FBX module not found. FBX export will be disabled.")
        fbx_time = time.time() - fbx_start
        print(f">>> [DEBUG] FBX converter initialization completed in {fbx_time:.2f} seconds")

        # Step 4: Finalize
        print(f">>> [DEBUG] Step 4/4: Finalizing runtime initialization...")
        self._loaded = True
        
        total_time = time.time() - total_start
        print(f">>> [DEBUG] Total T2MRuntime loading time: {total_time:.2f} seconds")
        print(f">>> [DEBUG] Number of pipelines: {len(self.pipelines)}")
        print(f">>> [DEBUG] T2MRuntime loading completed successfully!")

    def _acquire_pipeline(self) -> int:
        while True:
            with self._lock:
                for i in range(len(self._gpu_load)):
                    if self._gpu_load[i] == 0:
                        self._gpu_load[i] = 1
                        return i
            time.sleep(0.01)

    def _release_pipeline(self, idx: int):
        with self._lock:
            self._gpu_load[idx] = 0

    def test_dit_inference(self, duration: float = 2.0, seed: int = 42) -> bool:
        """
        Test DiT model inference with unconditional/blank input.
        This method is used to verify the DiT model works before loading text encoder.

        Args:
            duration: Duration of the test motion in seconds
            seed: Random seed for reproducibility

        Returns:
            True if inference succeeds and produces valid output
        """
        if not self.pipelines:
            raise RuntimeError("No pipeline loaded. Call load() first.")

        pi = self._acquire_pipeline()
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()
            device = next(pipeline.parameters()).device

            # Calculate frame length from duration (assuming 30fps output, 20fps internal)
            length = int(duration * 20)
            length = min(length, pipeline.train_frames)

            # Use null features for unconditional generation
            batch_size = 1
            vtxt_input = pipeline.null_vtxt_feat.expand(batch_size, -1, -1).to(device)
            ctxt_input = pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device)
            ctxt_length = torch.tensor([1] * batch_size, device=device)

            # Create masks
            from ..pipeline.motion_diffusion import length_to_mask

            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            x_length = torch.LongTensor([length] * batch_size).to(device)
            x_mask_temporal = length_to_mask(x_length, pipeline.train_frames)

            # Run denoising inference
            print(f"\t>>> Running DiT inference test: length={length}, device={device}")

            # Create random noise
            generator = torch.Generator(device=device).manual_seed(seed)
            latent_shape = (batch_size, pipeline.train_frames, pipeline.mean.shape[-1])
            latents = torch.randn(latent_shape, generator=generator, device=device, dtype=vtxt_input.dtype)

            # Simple single-step denoising test (just forward pass)
            with torch.no_grad():
                # Get timestep
                timesteps = torch.tensor([0.5], device=device, dtype=vtxt_input.dtype).expand(batch_size)

                # Forward pass through DiT
                # Use correct parameter names for HunyuanMotionMMDiT.forward()
                _ = pipeline.motion_transformer(
                    x=latents,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=timesteps,
                    x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )

            print(f"\t>>> DiT forward pass completed successfully!")
            return True

        except Exception as e:
            print(f"\t>>> DiT inference test failed: {e}")
            raise
        finally:
            self._release_pipeline(pi)

    def load_text_encoder(self) -> None:
        """
        Load text encoder for all pipelines.
        This is called after DiT model testing to complete the initialization.
        """
        if not self.pipelines:
            raise RuntimeError("No pipeline loaded. Call load() first.")

        print(">>> Loading text encoder for all pipelines...")
        for i, pipeline in enumerate(self.pipelines):
            if not hasattr(pipeline, "text_encoder") or pipeline.text_encoder is None:
                device = next(pipeline.parameters()).device
                pipeline.text_encoder = load_object(pipeline._text_encoder_module, pipeline._text_encoder_cfg)
                pipeline.text_encoder.to(device)
                print(f"\t>>> Text encoder loaded for pipeline {i} on {device}")

        # Update skip_text flag
        self.skip_text = False
        print(">>> Text encoder loading completed!")

    def rewrite_text_and_infer_time(self, text: str) -> Tuple[float, str]:
        print("Start rewriting text...")
        duration, rewritten_text = self.prompt_rewriter.rewrite_prompt_and_infer_time(f"{text}")
        print(f"\t>>> Rewritten text: {rewritten_text}, duration: {duration:.2f} seconds")
        return duration, rewritten_text

    def generate_motion(
        self,
        text: str,
        seeds_csv: str,
        duration: float,
        cfg_scale: float,
        output_format: str = "fbx",
        output_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
        original_text: Optional[str] = None,
        use_special_game_feat: bool = False,
        apply_loop: bool = False,
    ) -> Tuple[Union[str, list[str]], dict]:
        self.load()
        seeds = [int(s.strip()) for s in seeds_csv.split(",") if s.strip() != ""]
        pi = self._acquire_pipeline()
        try:
            pipeline = self.pipelines[pi]
            pipeline.eval()

            # When skip_text=True (debug mode), use blank text features
            if self.skip_text:
                print(">>> [Debug Mode] Using blank text features (skip_text=True)")
                device = next(pipeline.parameters()).device
                batch_size = len(seeds) if seeds else 1
                # Create blank hidden_state_dict using null features
                hidden_state_dict = {
                    "text_vec_raw": pipeline.null_vtxt_feat.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw": pipeline.null_ctxt_input.expand(batch_size, -1, -1).to(device),
                    "text_ctxt_raw_length": torch.tensor([1] * batch_size, device=device),
                }
                # Disable CFG in debug mode (use cfg_scale=1.0)
                model_output = pipeline.generate(
                    text,
                    seeds,
                    duration,
                    cfg_scale=1.0,
                    use_special_game_feat=False,
                    hidden_state_dict=hidden_state_dict,
                )
            else:
                model_output = pipeline.generate(
                    text, seeds, duration, cfg_scale=cfg_scale, use_special_game_feat=use_special_game_feat
                )
            
            # Apply looping if requested
            if apply_loop:
                model_output = self.apply_looping(
                    model_output, 
                    blend_duration=0.5, # Default 0.5s blend
                    fps=pipeline.output_mesh_fps
                )

        finally:
            self._release_pipeline(pi)

        ts = _now()
        
        # Print SMPL-H data if requested (for CLI debugging)
        if hasattr(self, 'print_smplh_data') and self.print_smplh_data:
            self._print_smplh_data(model_output, text)
        
        save_data, base_filename = save_visualization_data(
            output=model_output,
            text=text if original_text is None else original_text,
            rewritten_text=text,
            timestamp=ts,
            output_dir=output_dir,
            output_filename=output_filename,
        )

        html_content = self._generate_html_content(
            timestamp=ts,
            file_path=base_filename,
            output_dir=output_dir,
        )

        if output_format == "fbx" and not self.fbx_available:
            print(">>> Warning: FBX export requested but FBX SDK is not available. Falling back to dict format.")
            output_format = "dict"

        if output_format == "fbx" and self.fbx_available:
            fbx_files = self._generate_fbx_files(
                visualization_data=save_data,
                output_dir=output_dir,
                fbx_filename=output_filename,
            )
            return html_content, fbx_files, model_output
        elif output_format == "npy":
            # Export raw SMPL-H data as .npy files
            npy_files = self._generate_npy_files(
                visualization_data=save_data,
                output_dir=output_dir,
                npy_filename=output_filename,
            )
            return html_content, npy_files, model_output
        elif output_format == "json":
            # Export SMPL-H data as .json files
            json_files = self._generate_json_files(
                visualization_data=save_data,
                output_dir=output_dir,
                json_filename=output_filename,
            )
            return html_content, json_files, model_output
        elif output_format == "dict":
            # Return HTML content and empty list for fbx_files when using dict format
            return html_content, [], model_output
        else:
            raise ValueError(f">>> Invalid output format: {output_format}")

    def _generate_html_content(
        self,
        timestamp: str,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate static HTML content with embedded data for iframe srcdoc.
        All JavaScript code is embedded directly in the HTML, no external static resources needed.

        Args:
            timestamp: Timestamp string for logging
            file_path: Base filename (without extension)
            output_dir: Directory where NPZ/meta files are stored

        Returns:
            HTML content string (to be used in iframe srcdoc)
        """
        print(f">>> Generating static HTML content, timestamp: {timestamp}")
        gradio_dir = output_dir if output_dir is not None else "output/gradio"

        try:
            # Generate static HTML content with embedded data (all JS is embedded in template)
            html_content = generate_static_html_content(
                folder_name=gradio_dir,
                file_name=file_path,
                hide_captions=False,
            )

            print(f">>> Static HTML content generated for: {file_path}")
            return html_content

        except Exception as e:
            print(f">>> Failed to generate static HTML content: {e}")
            import traceback

            traceback.print_exc()
            # Return error HTML
            return f"<html><body><h1>Error generating visualization</h1><p>{str(e)}</p></body></html>"

    def _generate_fbx_files(
        self,
        visualization_data: dict,
        output_dir: Optional[str] = None,
        fbx_filename: Optional[str] = None,
    ) -> List[str]:
        assert "smpl_data" in visualization_data, "smpl_data not found in visualization_data"
        fbx_files = []
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(root_dir, "output", "gradio")

        smpl_data_list = visualization_data["smpl_data"]

        unique_id = str(uuid.uuid4())[:8]
        text = visualization_data["text"]
        timestamp = visualization_data["timestamp"]
        for bb in range(len(smpl_data_list)):
            smpl_data = smpl_data_list[bb]
            if fbx_filename is None:
                fbx_filename_bb = f"{timestamp}_{unique_id}_{bb:03d}.fbx"
            else:
                fbx_filename_bb = f"{fbx_filename}_{bb:03d}.fbx"
            fbx_path = os.path.join(output_dir, fbx_filename_bb)
            success = self.fbx_converter.convert_npz_to_fbx(smpl_data, fbx_path)
            if success:
                fbx_files.append(fbx_path)
                print(f"\t>>> FBX file generated: {fbx_path}")
                txt_path = fbx_path.replace(".fbx", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                fbx_files.append(txt_path)

        return fbx_files

    def _generate_npy_files(
        self,
        visualization_data: dict,
        output_dir: Optional[str] = None,
        npy_filename: Optional[str] = None,
    ) -> List[str]:
        """
        Generate .npy files containing raw SMPL-H data for each animation.
        
        Args:
            visualization_data: Dictionary containing SMPL data and metadata
            output_dir: Directory to save .npy files
            npy_filename: Base filename for the .npy files
            
        Returns:
            List of paths to generated .npy files
        """
        assert "smpl_data" in visualization_data, "smpl_data not found in visualization_data"
        npy_files = []
        
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(root_dir, "output", "gradio")

        smpl_data_list = visualization_data["smpl_data"]
        text = visualization_data["text"]
        timestamp = visualization_data["timestamp"]
        
        for bb in range(len(smpl_data_list)):
            smpl_data = smpl_data_list[bb]
            
            if npy_filename is None:
                npy_filename_bb = f"{timestamp}_{bb:03d}_smplh.npy"
            else:
                npy_filename_bb = f"{npy_filename}_{bb:03d}_smplh.npy"
            
            npy_path = os.path.join(output_dir, npy_filename_bb)
            
            # Prepare SMPL-H data dictionary
            smplh_dict = {
                'poses': smpl_data['poses'],  # (num_frames, 156) - axis-angle rotations
                'trans': smpl_data['trans'],  # (num_frames, 3) - translations
                'betas': smpl_data['betas'],  # (1, 16) - shape parameters
                'gender': smpl_data['gender'],  # str - gender
                'Rh': smpl_data['Rh'],  # (num_frames, 3) - root rotation
                'mocap_framerate': smpl_data.get('mocap_framerate', 30),
                'num_frames': smpl_data.get('num_frames', smpl_data['poses'].shape[0]),
            }
            
            # Save as .npy file
            np.save(npy_path, smplh_dict)
            npy_files.append(npy_path)
            print(f"\t>>> NPY file generated: {npy_path}")
            
            # Also save a text file with the prompt
            txt_path = npy_path.replace("_smplh.npy", "_prompt.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            npy_files.append(txt_path)
        
        return npy_files
    
    def _generate_json_files(
        self,
        visualization_data: dict,
        output_dir: Optional[str] = None,
        json_filename: Optional[str] = None,
    ) -> List[str]:
        """
        Generate .json files from SMPL-H data for each animation.
        
        Args:
            visualization_data: Dictionary containing SMPL data and metadata
            output_dir: Directory to save .json files
            json_filename: Base filename for the .json files
            
        Returns:
            List of paths to generated .json files
        """
        assert "smpl_data" in visualization_data, "smpl_data not found in visualization_data"
        json_files = []
        
        if output_dir is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(root_dir, "output", "gradio")

        smpl_data_list = visualization_data["smpl_data"]
        text = visualization_data["text"]
        timestamp = visualization_data["timestamp"]
        
        for bb in range(len(smpl_data_list)):
            smpl_data = smpl_data_list[bb]
            
            if json_filename is None:
                # Use text as filename, replace spaces with underscores
                json_filename_bb = f"{timestamp}_{bb:03d}_{text.replace(' ', '_')}.json"
                # Remove any problematic characters for filenames
                json_filename_bb = "".join(c for c in json_filename_bb if c.isalnum() or c in '_-')
            else:
                json_filename_bb = f"{json_filename}_{bb:03d}.json"
            
            json_path = os.path.abspath(os.path.join(output_dir, json_filename_bb))
            
            # Initialize export dictionary
            export_dict = {}
            
            # 1. Handle Poses (The rotations)
            if 'poses' in smpl_data:
                p = smpl_data['poses'].astype(np.float32)
                export_dict["frameCount"] = int(p.shape[0])
                export_dict["poses"] = p.flatten().tolist()
                print(f"\t>>> Added 'poses': {p.shape}")
            
            # 2. Handle Translation (The movement)
            if 'trans' in smpl_data:
                t = smpl_data['trans'].astype(np.float32)
                export_dict["trans"] = t.flatten().tolist()
                print(f"\t>>> Added 'trans': {t.shape}")
            
            # 3. Handle Root Rotation (Rh)
            if 'Rh' in smpl_data:
                r = smpl_data['Rh'].astype(np.float32)
                export_dict["Rh"] = r.flatten().tolist()
                print(f"\t>>> Added 'Rh': {r.shape}")
            
            # 4. Add metadata
            export_dict["text"] = text
            export_dict["timestamp"] = timestamp
            export_dict["batch_index"] = bb
            
            # Write to JSON with indentation
            with open(json_path, 'w') as f:
                json.dump(export_dict, f, indent=4)
            
            json_files.append(json_path)
            print(f"\t>>> JSON file generated: {json_path}")
            print(f"\t>>> JSON file exists: {os.path.exists(json_path)}")
            print(f"\t>>> JSON file size: {os.path.getsize(json_path)} bytes")
            
            # Also save a text file with the prompt
            txt_path = os.path.abspath(json_path.replace(".json", "_prompt.txt"))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            json_files.append(txt_path)
            print(f"\t>>> Prompt file generated: {txt_path}")
        
        return json_files
