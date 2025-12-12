import torch
import numpy as np
from comfy import model_management as mm
import math
import gc
from tqdm import tqdm
import torch.nn.functional as F
from comfy.utils import ProgressBar, common_upscale

from .framepack_helpers import (
    BenchmarkManager,
    PromptHandler,
    SchedulerFactory,
    RoPEEmbeddings,
    VAEProcessor,
    ContextBuilder,
    MaskGenerator,
    FrequencyProcessor,
    ReferenceImageProcessor,
    VAE_STRIDE
)
import time
from diffusers.schedulers import DEISMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler



class WanVACEVideoFramepackSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["dpm++", "unipc", "euler", "deis", "lcm"], {"default": "unipc"}),
                "mode": (["moc", "sparse", "frame", "all"], {"default": "moc"}),
                "num_frames": ("INT", {"default": 81, "min": 41, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "multi_prompts": ("STRING", {
                    "default": "A person walking in a park\nThe person starts jogging\nThe person runs faster\nThe person slows down to rest", 
                    "multiline": True
                }),
                "encode_prompts": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                "ref_images": ("IMAGE",),
                "input_frames": ("VIDEO",),
                "input_mask": ("MASK",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "text_embeds_list": ("LIST",),
                "wan_t5_model": ("WANTEXTENCODER",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "VIDEO")
    RETURN_NAMES = ("samples", "decoded_video")
    FUNCTION = "process"
    CATEGORY = "framepackVACE"
    DESCRIPTION = "A sampler specifically for the FramePack algorithm for long video generation using hierarchical context."

    def __init__(self):
        self.cache_state = None
        self.benchmark_manager = BenchmarkManager()
        
        # Optimize CUDA performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def process(self, model, vae, steps, cfg, shift, seed, scheduler, mode,
                num_frames, width, height, force_offload, multi_prompts,
                encode_prompts=True, ref_images=None, input_frames=None, 
                input_mask=None, negative_prompt="", sigmas=None, 
                text_embeds_list=None, wan_t5_model=None):
        """Main processing function for ComfyUI with multi-prompt support"""
        
        enable_benchmarking = True
        benchmark_output_dir = "./benchmarks"
        
        text_encoder = wan_t5_model
        device = mm.get_torch_device()
        self.device = device
        offload_device = mm.unet_offload_device()
        
        # Extract model components
        model_obj = model.model
        model_wrapper = model_obj.diffusion_model
        
        # Setup VAE
        self.vae_processor = VAEProcessor(vae.to(device).to(torch.float32), device)
        # model_wrapper.to(device) # Moved inside loop for multi-mode support
        
        # Ensure dimensions are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Calculate number of sections
        INITIAL_FRAMES = 81
        num_sections = 1 if num_frames <= INITIAL_FRAMES else math.ceil(num_frames / INITIAL_FRAMES)
        
        # Parse prompts
        section_prompts = PromptHandler.parse_multi_prompts(multi_prompts, num_sections)
        
        # Encode prompts
        if text_encoder is not None:
            print("Encoding prompts for each section...")
            section_text_embeds = []
            for i, prompt in enumerate(section_prompts):
                print(f"Encoding prompt {i+1}/{num_sections}: {prompt[:50]}...")
                text_embed = PromptHandler.encode_prompt_for_section(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    text_encoder=text_encoder,
                    device=device
                )
                section_text_embeds.append(text_embed)
        elif text_embeds_list:
            section_text_embeds = text_embeds_list
        else:
            raise ValueError("Either text encoder or pre-encoded embeddings required")
        
        # Determine modes to run
        modes_to_run = [mode] if mode != "all" else ["sparse", "frame", "moc"]
        generated_results = []
        
        for current_mode in modes_to_run:
            print(f"\n{'='*20}\nRunning Mode: {current_mode}\n{'='*20}")
            
            # Reset state
            self.cache_state = None
            
            # Clear internal model caches to ensure no state leakage
            if hasattr(model_wrapper, 'teacache_state'):
                model_wrapper.teacache_state.clear_all()
            if hasattr(model_wrapper, 'magcache_state'):
                model_wrapper.magcache_state.clear_all()
            if hasattr(model_wrapper, 'block_mask'):
                model_wrapper.block_mask = None
            
            # Full memory cleanup cycle to ensure fairness
            # 1. Move to CPU (unload from VRAM)
            model_wrapper.to(offload_device)
            # 2. Clear CUDA cache and Garbage Collect
            mm.soft_empty_cache()
            gc.collect()
            # 3. Move back to GPU (fresh load to VRAM)
            model_wrapper.to(device)
            
            # Initialize benchmarking for this run
            if enable_benchmarking:
                self.benchmark_manager = BenchmarkManager() # New instance for clean state
                self.benchmark_manager.overall_start_time = time.time()
                self.benchmark_manager.generation_params = {
                    'num_frames': num_frames,
                    'width': width,
                    'height': height,
                    'steps': steps,
                    'cfg': cfg,
                    'scheduler': scheduler,
                    'mode': current_mode,
                    'seed': seed,
                }
        
            # Generate video
            latents = self._generate_with_framepack_multi(
                model_wrapper=model_wrapper,
                section_text_embeds=section_text_embeds,
                section_prompts=section_prompts,
                input_frames=input_frames,
                input_masks=input_mask,
                ref_images=ref_images,
                width=width,
                height=height,
                num_frames=num_frames,
                shift=shift,
                scheduler_name=scheduler,
                mode=current_mode,
                steps=steps,
                cfg=cfg,
                seed=seed,
                sigmas=sigmas,
                device=device,
                offload_device=offload_device,
                force_offload=force_offload
            )
            
            generated_results.append(latents)
            
            # Generate and save benchmark report
            if enable_benchmarking:
                report = self.benchmark_manager.generate_report(section_prompts)
                print("\n" + report)
                self.benchmark_manager.save_report(report, benchmark_output_dir)
        
        # Combine results if multiple
        if len(generated_results) > 1:
            # Stack along batch dimension: [B, C, T, H, W]
            final_latents = torch.stack(generated_results, dim=0)
        else:
            final_latents = generated_results[0].unsqueeze(0)

        return ({"samples": final_latents}, )

    def _generate_with_framepack_multi(self, model_wrapper, section_text_embeds, 
                                       section_prompts, input_frames, input_masks, 
                                       ref_images, width, height, num_frames,
                                       shift, scheduler_name, mode, steps, cfg, seed, sigmas,
                                       device, offload_device, force_offload):
        """Core FramePack generation algorithm with multi-prompt support"""
        
        vae_dtype = torch.float32
        all_generated_latents = []
        accumulated_latents = []
        total_output_frames = 0

        LATENT_WINDOW = 41
        GENERATION_FRAMES = 30
        CONTEXT_FRAMES = 11
        INITIAL_FRAMES = 81
        
        # Initialize FramePackCompressor if in frame mode
        frame_compressor = None
        if mode == "frame":
            from .framepack_helpers import FramePackCompressor
            frame_compressor = FramePackCompressor()
            print("Initialized FramePackCompressor for frame-level compression")
        
        num_sections = 1 if num_frames <= INITIAL_FRAMES else math.ceil(num_frames / INITIAL_FRAMES)
        
        for section in range(num_sections):
            print(f"\n[Section {section+1}/{num_sections}]")
            print(f"Using prompt: {section_prompts[section][:100]}...")
            
            text_embeds = section_text_embeds[section]
            
            # PHASE 1: ENCODING
            self.benchmark_manager.benchmark_section(section, 'encoding')
            
            framepack_history = []
            vace_data = None
            
            if section == 0:
                # Initial section setup
                input_frames = torch.zeros(1, 3, INITIAL_FRAMES, height, width, 
                                          device=device, dtype=vae_dtype)
                input_masks = torch.ones_like(input_frames, device=device, dtype=vae_dtype)
                input_frames = [(f * 2 - 1) for f in input_frames]
                
                # Process reference images if provided
                if ref_images is not None:
                    ref_images = ReferenceImageProcessor.process_reference_images(
                        ref_images, width, height, device, vae_dtype
                    )
                
                target_shape = (
                    16,
                    (INITIAL_FRAMES - 1) // VAE_STRIDE[0] + 1,
                    height // VAE_STRIDE[1],
                    width // VAE_STRIDE[2]
                )
            else:
                # Build context based on mode
                if mode == "sparse":
                    # Legacy sparse selection
                    context_latent = ContextBuilder.build_hierarchical_context(accumulated_latents, section)
                    hierarchical_frames = ContextBuilder.pick_context(context_latent, section)
                    
                    # Decode to frames for re-encoding (legacy behavior)
                    # Note: This is inefficient but preserves exact legacy behavior
                    input_frames = self.vae_processor.decode_latent([hierarchical_frames], None)
                    input_frames[0] = input_frames[0].expand(3, -1, -1, -1)
                    
                    input_masks = MaskGenerator.create_temporal_blend_mask(
                        input_frames[0].shape, section, device
                    )
                    
                    # Encode to latent space
                    z0 = self.vae_processor.encode_frames(input_frames, ref_images=None,
                                                         masks=input_masks, tiled_vae=False)
                    m0 = self.vae_processor.encode_masks(input_masks, ref_images=None)
                    z = self.vae_processor.combine_latent(z0, m0)
                    
                    vace_data = [{
                        "context": z,
                        "scale": [1.0] * (steps + 1),
                        "start": 0.0,
                        "end": 1.0,
                        "seq_len": math.ceil((width // VAE_STRIDE[2] * height // VAE_STRIDE[1]) / 4 * LATENT_WINDOW)
                    }]
                    
                elif mode == "frame":
                    # Frame-level compression
                    # accumulated_latents contains latents [C, T, H, W]
                    # We need to compress them and select context
                    
                    # Compress history
                    compressed_history = frame_compressor.compress_history(accumulated_latents)
                    
                    # Select context frames (returns [C, 41, H, W])
                    context_latent = frame_compressor.select_context_frames(
                        compressed_history, 
                        num_context_frames=CONTEXT_FRAMES,
                        add_generation_frames=True
                    ).to(device)
                    
                    # Check for channel mismatch with model expectation (e.g. 16 vs 96)
                    if hasattr(model_wrapper, 'vace_in_dim') and model_wrapper.vace_in_dim != context_latent.shape[0]:
                        target_dim = model_wrapper.vace_in_dim
                        current_dim = context_latent.shape[0]
                        if target_dim > current_dim and target_dim % current_dim == 0:
                            repeat_factor = target_dim // current_dim
                            print(f"Adapting context latent channels for Frame mode: {current_dim} -> {target_dim} (repeat {repeat_factor}x)")
                            context_latent = context_latent.repeat(repeat_factor, 1, 1, 1)
                        else:
                            print(f"WARNING: Context latent channel mismatch in Frame mode: Got {current_dim}, Model expects {target_dim}. Cannot adapt automatically.")

                    # We treat this as the "input" for the model, but since we already have latents,
                    # we need to adapt how we pass it.
                    # The model expects noise input, and we usually encode frames to get z.
                    # Here we have z directly.
                    
                    # For consistency with the pipeline, we'll set vace_data with this context
                    # This mimics the sparse mode but with compressed frames
                    
                    # We need to wrap it in list as combine_latent does
                    z = [context_latent]
                    
                    vace_data = [{
                        "context": z,
                        "scale": [1.0] * (steps + 1),
                        "start": 0.0,
                        "end": 1.0,
                        "seq_len": math.ceil((width // VAE_STRIDE[2] * height // VAE_STRIDE[1]) / 4 * LATENT_WINDOW)
                    }]
                    
                else: # mode == "moc" (default)
                    # Build FramePack history for MoC
                    for l in accumulated_latents:
                        # l is [C, T, H, W]
                        # unsqueeze to [1, C, T, H, W]
                        framepack_history.append(l.unsqueeze(0))
                
                # Use standard target shape for subsequent sections
                target_shape = (
                    16,
                    (INITIAL_FRAMES - 1) // VAE_STRIDE[0] + 1,
                    height // VAE_STRIDE[1],
                    width // VAE_STRIDE[2]
                )
                
                ref_images = None
            
            self.benchmark_manager.benchmark_section(section, 'encoding')  # End encoding
            
            # PHASE 2: DENOISING
            self.benchmark_manager.benchmark_section(section, 'denoising')
            
            # Setup scheduler
            sample_scheduler = SchedulerFactory.create_scheduler(
                scheduler_name, steps, shift, device, sigmas
            )
            timesteps = sample_scheduler.timesteps
            
            # Initialize noise
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed if seed != -1 else torch.randint(0, 2**32, (1,)).item())
            
            has_ref = ref_images is not None
            noise = torch.randn(
                target_shape[0],
                target_shape[1] + (1 if has_ref else 0),
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device="cpu",
                generator=generator
            )
            
            latent = noise.to(device)
            
            # Setup model parameters
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
            freqs = RoPEEmbeddings.setup_rope_embeddings(model_wrapper, latent.shape[1])
            num_steps = len(timesteps)

            # Ensure cfg is a list
            if not isinstance(cfg, list):
                cfg = [cfg] * (steps + 1)
            
            # Setup progress bar
            pbar = ProgressBar(steps)
            
            # Clear memory before generation
            mm.soft_empty_cache()
            gc.collect()
            
            # Initialize cache state
            self.cache_state = [None, None]
            
            # Main denoising loop
            for idx, t in enumerate(timesteps):
                print(idx+1, 'of ',num_steps )
                timestep = torch.tensor([t]).to(device)
                
                # Get noise prediction
                noise_pred = self._predict_with_cfg(
                    latent=latent,
                    cfg_scale=cfg[idx],
                    text_embeds=text_embeds,
                    timestep=timestep,
                    idx=idx,
                    model_wrapper=model_wrapper,
                    vace_data=vace_data,
                    seq_len=seq_len,
                    freqs=freqs,
                    device=device,
                    framepack_history=framepack_history
                )
                
                # Scheduler step
                step_args = {"generator": generator}
                if isinstance(sample_scheduler, (DEISMultistepScheduler, FlowMatchScheduler)):
                    step_args.pop("generator", None)
                
                latent = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    **step_args
                )[0].squeeze(0)
                
                pbar.update(1)
                
                # Memory management
                if force_offload and idx % 10 == 0:
                    mm.soft_empty_cache()
            
            self.benchmark_manager.benchmark_section(section, 'denoising')  # End denoising
            
            # PHASE 3: ACCUMULATION
            self.benchmark_manager.benchmark_section(section, 'accumulation')
            
            # Handle accumulation based on section
            if section == 0:
                if ref_images is not None:
                    latent_without_ref = latent[:, 1:, :, :]
                else:
                    latent_without_ref = latent
                
                accumulated_latents.append(latent_without_ref)
                all_generated_latents.append(latent_without_ref)
            else:
                # Add new frames
                new = latent[:, -GENERATION_FRAMES:, :, :]
                accumulated_latents.append(new)
                
                # Add to final output (skip overlap frames)
                new_content = latent[:, -GENERATION_FRAMES:, :, :]
                new_content = new_content[:, CONTEXT_FRAMES:, :, :]
                all_generated_latents.append(new_content)
                
                frames_added = new_content.shape[1]
                total_output_frames += frames_added
                print(f"Added {frames_added} frames (total: {total_output_frames})")
            
            self.benchmark_manager.benchmark_section(section, 'accumulation')  # End accumulation
            
            # Clear cache after section
            if 'noise_pred' in locals():
                del latent, noise_pred
            mm.soft_empty_cache()
            gc.collect()
        
        # Move model to offload device if requested
        if force_offload:
            model_wrapper.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()
        
        final_latent = torch.cat(all_generated_latents, dim=1)
        return final_latent.cpu()

    def _predict_with_cfg(self, latent, cfg_scale, text_embeds, timestep, idx,
                         model_wrapper, vace_data, seq_len, freqs, device, framepack_history=None):
        """Classifier-free guidance prediction"""
        
        dtype = torch.float32
        latent = latent.to(dtype)
        
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype):
            # Prepare base parameters
            base_params = {
                'seq_len': seq_len,
                'device': device,
                'freqs': freqs,
                't': timestep.to(device),
                'current_step': idx,
                "nag_params": text_embeds.get("nag_params", {}),
                "nag_context": text_embeds.get("nag_prompt_embeds", None),
                "ref_target_masks": None,
            }

            # Check if model accepts framepack_history (handle reload issues)
            import inspect
            try:
                forward_method = getattr(model_wrapper, 'forward', None)
                if forward_method:
                    forward_params = inspect.signature(forward_method).parameters
                    if "framepack_history" in forward_params:
                        base_params["framepack_history"] = framepack_history
                    elif framepack_history is not None and idx == 0:
                        print("WARNING: WanModel does not accept 'framepack_history'. Please restart ComfyUI to reload the updated model definition.")
            except Exception as e:
                print(f"WARNING: Could not inspect model signature: {e}")
            
            current_step_percentage = idx / 30
            
            # Conditional prediction
            noise_pred_cond, cache_state_cond = model_wrapper(
                [latent],
                context=text_embeds["prompt_embeds"],
                y=None,
                clip_fea=None,
                is_uncond=False,
                current_step_percentage=current_step_percentage,
                pred_id=self.cache_state[0] if self.cache_state else None,
                vace_data=vace_data,
                attn_cond=None,
                **base_params
            )
            noise_pred_cond = noise_pred_cond[0]
            
            # If cfg_scale is 1.0, skip unconditional
            if math.isclose(cfg_scale, 1.0):
                self.cache_state = [cache_state_cond, None]
                return noise_pred_cond
            
            # Unconditional prediction
            noise_pred_uncond, cache_state_uncond = model_wrapper(
                [latent],
                context=text_embeds["negative_prompt_embeds"],
                y=None,
                clip_fea=None,
                is_uncond=True,
                current_step_percentage=current_step_percentage,
                pred_id=self.cache_state[1] if self.cache_state else None,
                vace_data=vace_data,
                attn_cond=None,
                **base_params
            )
            noise_pred_uncond = noise_pred_uncond[0]
            
            # Apply CFG
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Update cache state
            self.cache_state = [cache_state_cond, cache_state_uncond]
            
            return noise_pred


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": WanVACEVideoFramepackSampler2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": "WanVACE FramePack Sampler 2"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']