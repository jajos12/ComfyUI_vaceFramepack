"""
FramePack Helper Functions
Contains utility functions for video generation, encoding, context building, and benchmarking.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import gc
import json
import os
from datetime import datetime
from tqdm import tqdm
import psutil
from typing import List, Tuple, Optional

from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)

from comfy import model_management as mm
from comfy.utils import common_upscale

from .wanvideo.utils.rope import rope_params
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from .wanvideo.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from .wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler


# Constants
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


class BenchmarkManager:
    """Manages benchmarking and performance tracking"""
    
    def __init__(self):
        self.section_benchmarks = {}
        self.overall_start_time = None
        self.generation_params = {}
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        stats = {}

        # CPU memory (process)
        process = psutil.Process()
        stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        stats["cpu_memory_percent"] = process.memory_percent()

        # System memory
        mem = psutil.virtual_memory()
        stats["system_memory_total_gb"] = mem.total / 1024**3
        stats["system_memory_used_gb"] = mem.used / 1024**3
        stats["system_memory_percent"] = mem.percent

        # GPU stats (if CUDA available)
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

            try:
                nvmlInit()
                device_count = nvmlDeviceGetCount()
                if device_count > 0:
                    handle = nvmlDeviceGetHandleByIndex(0)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    stats["gpu_memory_total_gb"] = mem_info.total / 1024**3
                    stats["gpu_memory_used_gb"] = mem_info.used / 1024**3
                    util = nvmlDeviceGetUtilizationRates(handle)
                    stats["gpu_utilization_percent"] = util.gpu
            except Exception as e:
                print(f"GPU stats error: {e}")
            finally:
                try:
                    nvmlShutdown()
                except:
                    pass
        else:
            stats.update({
                "gpu_memory_allocated_gb": 0,
                "gpu_memory_reserved_gb": 0,
                "gpu_memory_total_gb": 0,
                "gpu_memory_used_gb": 0,
                "gpu_utilization_percent": 0,
            })

        return stats

    def benchmark_section(self, section_id, phase_name):
        """Start or end benchmarking for a section phase"""
        if section_id not in self.section_benchmarks:
            self.section_benchmarks[section_id] = {}
        
        phase_key = f"{phase_name}_start"
        phase_end_key = f"{phase_name}_end"
        
        if phase_key not in self.section_benchmarks[section_id]:
            # Starting phase
            self.section_benchmarks[section_id][phase_key] = time.time()
            self.section_benchmarks[section_id][f"{phase_name}_memory_start"] = self.get_memory_stats()
        else:
            # Ending phase
            self.section_benchmarks[section_id][phase_end_key] = time.time()
            self.section_benchmarks[section_id][f"{phase_name}_memory_end"] = self.get_memory_stats()
            
            # Calculate duration
            duration = self.section_benchmarks[section_id][phase_end_key] - self.section_benchmarks[section_id][phase_key]
            self.section_benchmarks[section_id][f"{phase_name}_duration"] = duration
            
            # Calculate memory delta
            start_mem = self.section_benchmarks[section_id][f"{phase_name}_memory_start"]
            end_mem = self.section_benchmarks[section_id][f"{phase_name}_memory_end"]
            
            if 'gpu_memory_allocated_gb' in start_mem and 'gpu_memory_allocated_gb' in end_mem:
                gpu_delta = end_mem['gpu_memory_allocated_gb'] - start_mem['gpu_memory_allocated_gb']
                self.section_benchmarks[section_id][f"{phase_name}_gpu_memory_delta_gb"] = gpu_delta
            
            cpu_delta = end_mem['cpu_memory_mb'] - start_mem['cpu_memory_mb']
            self.section_benchmarks[section_id][f"{phase_name}_cpu_memory_delta_mb"] = cpu_delta
    
    def generate_report(self, section_prompts=None):
        """Generate a comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("FRAMEPACK VIDEO GENERATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if self.overall_start_time:
            total_duration = time.time() - self.overall_start_time
            report.append(f"Total Processing Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        if self.generation_params:
            report.append("\nGeneration Parameters:")
            for key, value in self.generation_params.items():
                report.append(f"  {key}: {value}")
        
        report.append("\n" + "-" * 80)
        report.append("SECTION-BY-SECTION BREAKDOWN")
        report.append("-" * 80)
        
        total_encoding_time = 0
        total_denoising_time = 0
        total_accumulation_time = 0
        
        for section_id in sorted(self.section_benchmarks.keys()):
            section_data = self.section_benchmarks[section_id]
            report.append(f"\n[Section {section_id + 1}]")
            
            # Prompt info
            if section_prompts and section_id < len(section_prompts):
                report.append(f"Prompt: {section_prompts[section_id][:50]}...")
            
            # Timing breakdown
            phases = ['encoding', 'denoising', 'accumulation']
            for phase in phases:
                if f"{phase}_duration" in section_data:
                    duration = section_data[f"{phase}_duration"]
                    report.append(f"  {phase.capitalize()}: {duration:.2f}s")
                    
                    if phase == 'encoding':
                        total_encoding_time += duration
                    elif phase == 'denoising':
                        total_denoising_time += duration
                    elif phase == 'accumulation':
                        total_accumulation_time += duration
                    
                    # Memory changes
                    if f"{phase}_gpu_memory_delta_gb" in section_data:
                        gpu_delta = section_data[f"{phase}_gpu_memory_delta_gb"]
                        report.append(f"    GPU Memory Δ: {gpu_delta:+.3f} GB")
                    
                    if f"{phase}_cpu_memory_delta_mb" in section_data:
                        cpu_delta = section_data[f"{phase}_cpu_memory_delta_mb"]
                        report.append(f"    CPU Memory Δ: {cpu_delta:+.1f} MB")
            
            # Per-section total
            section_total = sum([section_data.get(f"{p}_duration", 0) for p in phases])
            report.append(f"  Section Total: {section_total:.2f}s")
            
            # Peak memory for section
            if 'denoising_memory_end' in section_data:
                end_mem = section_data['denoising_memory_end']
                if 'gpu_memory_allocated_gb' in end_mem:
                    report.append(f"  Peak GPU Memory: {end_mem['gpu_memory_allocated_gb']:.3f} GB")
                report.append(f"  Peak CPU Memory: {end_mem['cpu_memory_mb']:.1f} MB")
        
        # Summary statistics
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)
        
        num_sections = len(self.section_benchmarks)
        report.append(f"Total Sections Processed: {num_sections}")
        report.append(f"Total Encoding Time: {total_encoding_time:.2f}s")
        report.append(f"Total Denoising Time: {total_denoising_time:.2f}s")
        report.append(f"Total Accumulation Time: {total_accumulation_time:.2f}s")
        
        if num_sections > 0:
            report.append(f"Average Time per Section: {(total_encoding_time + total_denoising_time + total_accumulation_time) / num_sections:.2f}s")
            report.append(f"Average Denoising Time per Section: {total_denoising_time / num_sections:.2f}s")
        
        # Final memory state
        final_memory = self.get_memory_stats()
        report.append(f"\nFinal Memory State:")
        if 'gpu_memory_allocated_gb' in final_memory:
            report.append(f"  GPU Memory: {final_memory['gpu_memory_allocated_gb']:.3f} GB allocated")
        report.append(f"  CPU Memory: {final_memory['cpu_memory_mb']:.1f} MB")
        report.append(f"  System Memory: {final_memory['system_memory_percent']:.1f}% used")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report, output_dir="./benchmarks"):
        """Save benchmark report to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"framepack_benchmark_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Also save as JSON for easier analysis
        json_filename = f"framepack_benchmark_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        benchmark_dict = {
            'timestamp': timestamp,
            'generation_params': self.generation_params,
            'section_benchmarks': self.section_benchmarks,
            'total_duration': time.time() - self.overall_start_time if self.overall_start_time else 0
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(benchmark_dict, f, indent=2, default=str)
        
        print(f"Benchmark report saved to: {filepath}")
        print(f"JSON data saved to: {json_filepath}")
        
        return filepath


class PromptHandler:
    """Handles prompt parsing and encoding"""
    
    @staticmethod
    def parse_multi_prompts(multi_prompts, num_sections):
        """
        Parse multi-line prompts and assign them to sections.
        Each line represents a prompt for a section.
        If fewer prompts than sections, the last prompt is repeated.
        """
        # Split by newline and filter empty lines
        prompts = [p.strip() for p in multi_prompts.split('\n') if p.strip()]
        
        if not prompts:
            raise ValueError("No prompts provided in multi_prompts")
        
        # Assign prompts to sections
        section_prompts = []
        for section in range(num_sections):
            if section < len(prompts):
                section_prompts.append(prompts[section])
            else:
                # Repeat the last prompt for remaining sections
                section_prompts.append(prompts[-1])
        
        print(f"Parsed {len(prompts)} unique prompts for {num_sections} sections")
        for i, prompt in enumerate(section_prompts):
            print(f"  Section {i}: {prompt[:50]}...")
        
        return section_prompts
    
    @staticmethod
    def parse_prompt_weights(prompt):
        """
        Parse prompt weights in the format (text:weight).
        Returns cleaned prompt and weight dictionary.
        """
        import re
        
        weights = {}
        cleaned_prompt = prompt
        
        # Pattern to find (text:weight) format
        pattern = r'\(([^:)]+):([0-9.]+)\)'
        matches = re.findall(pattern, prompt)
        
        for text, weight_str in matches:
            try:
                weight = float(weight_str)
                weights[text.strip()] = weight
                # Remove the weight notation from the prompt
                cleaned_prompt = cleaned_prompt.replace(f"({text}:{weight_str})", text)
            except ValueError:
                print(f"Invalid weight value: {weight_str}")
        
        return cleaned_prompt.strip(), weights
    
    @staticmethod
    def encode_prompt_for_section(prompt, negative_prompt, text_encoder, device):
        """
        Encode a single prompt for a section using the WAN Video text encoder.
        Supports weighted prompts using (text:weight) syntax.
        """
        if text_encoder is None:
            raise ValueError("Text encoder is required for encoding prompts")
        
        # Extract the encoder model and dtype
        encoder = text_encoder["model"]
        dtype = text_encoder["dtype"]
        
        # Split positive prompts by '|' and process weights
        positive_prompts_raw = [p.strip() for p in prompt.split('|')]
        positive_prompts = []
        all_weights = []
        
        for p in positive_prompts_raw:
            cleaned_prompt, weights = PromptHandler.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)
        
        # Move encoder to device
        encoder.model.to(device)
        
        try:
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
                # Encode positive and negative prompts
                context = encoder(positive_prompts, device)
                context_null = encoder([negative_prompt if negative_prompt else ""], device)
                
                # Apply weights to embeddings if any were extracted
                for i, weights in enumerate(all_weights):
                    if weights:  # Only apply if weights exist
                        for text, weight in weights.items():
                            print(f"Applying weight {weight} to prompt: {text}")
                            context[i] = context[i] * weight
        finally:
            # Always move encoder back to CPU to free VRAM
            encoder.model.to('cpu')
            mm.soft_empty_cache()
        
        # Create the embedding dictionary with all required fields for WAN Video
        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
        }
        
        return prompt_embeds_dict


class SchedulerFactory:
    """Factory for creating schedulers"""
    
    @staticmethod
    def create_scheduler(scheduler_name, steps, shift, device, sigmas=None):
        """Setup the appropriate scheduler"""
        
        if scheduler_name == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type="dpmsolver++")
            if sigmas is None:
                scheduler.set_timesteps(steps, device=device)
            else:
                scheduler.sigmas = sigmas.to(device)
                scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                scheduler.num_inference_steps = len(scheduler.timesteps)
                
        elif scheduler_name == "unipc":
            scheduler = FlowUniPCMultistepScheduler(shift=shift)
            if sigmas is None:
                scheduler.set_timesteps(steps, device=device, shift=shift)
            else:
                scheduler.sigmas = sigmas.to(device)
                scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                scheduler.num_inference_steps = len(scheduler.timesteps)
                
        elif scheduler_name == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            scheduler.set_timesteps(steps, device=device, sigmas=sigmas.tolist() if sigmas else None)
            
        elif scheduler_name == "deis":
            scheduler = DEISMultistepScheduler(
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                flow_shift=shift
            )
            scheduler.set_timesteps(steps, device=device)
            scheduler.sigmas[-1] = 1e-6
            
        elif scheduler_name == "lcm":
            scheduler = FlowMatchLCMScheduler(shift=shift)
            scheduler.set_timesteps(steps, device=device, sigmas=sigmas.tolist() if sigmas else None)
            
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return scheduler


class RoPEEmbeddings:
    """Handles RoPE embeddings setup"""
    
    @staticmethod
    def setup_rope_embeddings(model_wrapper, latent_video_length):
        """Setup RoPE embeddings for the model"""
        
        model_wrapper.rope_embedder.k = None
        model_wrapper.rope_embedder.num_frames = None
        
        d = model_wrapper.dim // model_wrapper.num_heads
        riflex_freq_index = 0
        
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        
        return freqs


class VAEProcessor:
    """Handles VAE encoding and decoding operations"""
    
    def __init__(self, vae, device):
        self.vae = vae
        self.device = device
    
    def encode_frames(self, frames, ref_images, masks=None, tiled_vae=False):
        """Encode frames to latent space"""
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames, device=self.device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, device=self.device, tiled=tiled_vae)
            reactive = self.vae.encode(reactive, device=self.device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
        
        self.vae.model.clear_cache()
        cat_latents = []
        
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                else:
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        
        return cat_latents

    def encode_masks(self, masks, ref_images=None):
        """Encode masks to latent space"""
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )
            mask = mask.permute(2, 4, 0, 1, 3)
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        
        return result_masks
    
    def combine_latent(self, z, m):
        """Combine latents and masks"""
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def decode_latent(self, zs, ref_images=None):
        """Decode latents back to frames"""
        return self.vae.decode(zs, device=self.device)


class ContextBuilder:
    """Handles hierarchical context building and frame selection"""
    
    @staticmethod
    def build_hierarchical_context(accumulated_latents, section_id):
        """Build hierarchical context from accumulated latents"""
        if not accumulated_latents:
            raise ValueError("No accumulated latents available")

        all_prev = torch.cat(accumulated_latents, dim=1)
        total_frames = all_prev.shape[1]

        print(f"Building context from {total_frames} accumulated frames")

        return all_prev
    
    @staticmethod
    def pick_context(frames, section_id, initial=False):
        """
        Enhanced hierarchical context selection with constant 41-frame output.
        """
        # Constants
        LONG_FRAMES = 5
        MID_FRAMES = 3
        RECENT_FRAMES = 1
        OVERLAP_FRAMES = 2
        GEN_FRAMES = 30
        TOTAL_FRAMES = 41

        C, T, H, W = frames.shape

        if initial and T == TOTAL_FRAMES:
            return frames

        if initial and T < TOTAL_FRAMES:
            padding_needed = TOTAL_FRAMES - T
            padding = torch.zeros((C, padding_needed, H, W), device=frames.device)
            return torch.cat([frames, padding], dim=1)

        selected_indices = []

        # Long-term context
        if T >= 40:
            step = max(4, T // 20)
            long_indices = []
            for i in range(LONG_FRAMES):
                idx = min(i * step, T - 15)
                long_indices.append(idx)
            selected_indices.extend(long_indices)
        else:
            if T >= LONG_FRAMES:
                step = T // LONG_FRAMES
                long_indices = [i * step for i in range(LONG_FRAMES)]
            else:
                long_indices = list(range(T))
                while len(long_indices) < LONG_FRAMES:
                    long_indices.append(T - 1)
            selected_indices.extend(long_indices[:LONG_FRAMES])

        # Mid-term context
        mid_start = max(LONG_FRAMES, T - 15)
        mid_indices = [
            min(mid_start, T - 1),
            min(mid_start + 2, T - 1)
        ]
        selected_indices.extend(mid_indices)

        # Recent context
        recent_idx = max(0, T - 5)
        selected_indices.append(recent_idx)

        # Overlap frames
        overlap_start = max(0, T - OVERLAP_FRAMES)
        overlap_indices = list(range(overlap_start, T))
        while len(overlap_indices) < OVERLAP_FRAMES:
            overlap_indices.append(T - 1)
        selected_indices.extend(overlap_indices[:OVERLAP_FRAMES])

        context_frames = frames[:, selected_indices, :, :]
        gen_placeholder = torch.zeros((C, GEN_FRAMES, H, W), device=frames.device)

        final_frames = torch.cat([
            context_frames[:, :LONG_FRAMES],
            context_frames[:, LONG_FRAMES:LONG_FRAMES+MID_FRAMES],
            context_frames[:, LONG_FRAMES+MID_FRAMES:LONG_FRAMES+MID_FRAMES+RECENT_FRAMES],
            context_frames[:, -OVERLAP_FRAMES:],
            gen_placeholder
        ], dim=1)

        assert final_frames.shape[1] == TOTAL_FRAMES, \
            f"Expected {TOTAL_FRAMES} frames, got {final_frames.shape[1]}"

        if section_id % 5 == 0:
            print(f"\nContext selection debug (section {section_id}):")
            print(f"  Input frames: {T}")
            print(f"  Selected indices: {selected_indices}")
            print(f"  Output shape: {final_frames.shape}")

        return final_frames


class MaskGenerator:
    """Handles mask generation for temporal blending"""
    
    @staticmethod
    def create_temporal_blend_mask(frame_shape, section_id, device, initial=False):
        """Enhanced mask creation that handles decoded frame dimensions"""
        C, T, H, W = frame_shape
        
        # Constants
        LATENT_FRAMES = 41
        decoded_frames = T
        expansion_ratio = decoded_frames / LATENT_FRAMES
        
        mask = torch.zeros(3, decoded_frames, H, W, device=device)
        
        # Scale all frame counts by the expansion ratio
        LONG_FRAMES = int(5 * expansion_ratio)
        MID_FRAMES = int(3 * expansion_ratio)
        RECENT_FRAMES = int(1 * expansion_ratio)
        OVERLAP_FRAMES = int(2 * expansion_ratio)
        GEN_FRAMES = decoded_frames - (LONG_FRAMES + MID_FRAMES + RECENT_FRAMES + OVERLAP_FRAMES)
        
        if initial:
            mask[:, :-GEN_FRAMES] = 0.0
            mask[:, -GEN_FRAMES:] = 1.0
            return [mask]
        
        # Apply mask values
        idx = 0
        mask[:, idx:idx+LONG_FRAMES] = 0.05
        idx += LONG_FRAMES
        
        mask[:, idx:idx+MID_FRAMES] = 0.2
        idx += MID_FRAMES
        
        mask[:, idx:idx+RECENT_FRAMES] = 0.3
        idx += RECENT_FRAMES
        
        for i in range(OVERLAP_FRAMES):
            blend_value = 0.4 + (i / (OVERLAP_FRAMES - 1)) * 0.4
            mask[:, idx+i] = blend_value
        idx += OVERLAP_FRAMES
        
        mask[:, idx:] = 1.0
        
        return [mask]
    
    @staticmethod
    def create_spatial_variation(H, W, device):
        """Create spatial variation mask for natural blending"""
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        distance = torch.sqrt(x_grid**2 + y_grid**2) / 1.414
        variation = 1.0 - 0.3 * torch.exp(-3 * distance**2)

        return variation


class FrequencyProcessor:
    """Handles frequency domain operations"""
    
    @staticmethod
    def separate_appearance_and_motion(frames):
        """Use frequency domain to separate appearance from motion"""
        C, T, H, W = frames.shape

        # FFT
        fft_frames = torch.fft.rfft2(frames, dim=(-2, -1))
        fft_h = H
        fft_w = W // 2 + 1

        h_freqs = torch.fft.fftfreq(H, device=frames.device)
        w_freqs = torch.fft.rfftfreq(W, device=frames.device)
        h_grid, w_grid = torch.meshgrid(h_freqs, w_freqs, indexing='ij')

        freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)
        cutoff = 0.1
        low_pass_mask = (freq_magnitude < cutoff).float().to(frames.device)

        if low_pass_mask.shape != fft_frames.shape[-2:]:
            low_pass_mask = low_pass_mask[:fft_h, :fft_w]

        while low_pass_mask.dim() < fft_frames.dim():
            low_pass_mask = low_pass_mask.unsqueeze(0)

        appearance_fft = fft_frames * low_pass_mask
        motion_fft = fft_frames * (1 - low_pass_mask)

        appearance = torch.fft.irfft2(appearance_fft, s=(H, W))
        motion = torch.fft.irfft2(motion_fft, s=(H, W))

        return appearance, motion


class ReferenceImageProcessor:
    """Handles reference image processing"""
    
    @staticmethod
    def process_reference_images(ref_images, width, height, device, dtype):
        """Process reference images for generation"""
        if ref_images.shape[0] > 1:
            ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], 
                                dim=1).unsqueeze(0)
    
        B, H, W, C = ref_images.shape
        current_aspect = W / H
        target_aspect = width / height
        
        if current_aspect > target_aspect:
            new_h = int(W / target_aspect)
            pad_h = (new_h - H) // 2
            padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], 
                            device=ref_images.device, dtype=ref_images.dtype)
            padded[:, pad_h:pad_h+H, :, :] = ref_images
            ref_images = padded
        elif current_aspect < target_aspect:
            new_w = int(H * target_aspect)
            pad_w = (new_w - W) // 2
            padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], 
                            device=ref_images.device, dtype=ref_images.dtype)
            padded[:, :, pad_w:pad_w+W, :] = ref_images
            ref_images = padded
            
        ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, 
                                "lanczos", "center").movedim(1, -1)
        ref_images = ref_images.to(dtype).to(device).unsqueeze(0)
        ref_images = ref_images.permute(0, 4, 1, 2, 3).unsqueeze(0)
        ref_images = ref_images * 2 - 1
        
        return ref_images


class FramePackManager:
    """
    Manages FramePack compression logic, scheduling, and RoPE generation.
    Implements 'Dense Compression' strategy.
    """
    
    # Compression schedule configuration
    # Format: (distance_start, distance_end, compression_factor)
    # Distance is from the start of the generation window backwards
    SCHEDULE_CONFIG = [
        (0, 5, 1),      # Recent: Full resolution (1x2x2 effective)
        (5, 20, 4),     # Mid-term: 4x compression (1x4x4 effective)
        (20, 50, 16),   # Long-term: 16x compression (2x8x8 effective)
        (50, float('inf'), 64) # Deep history: 64x compression (4x16x16 effective)
    ]
    
    # Base patch size of the model (t, h, w)
    BASE_PATCH_SIZE = (1, 2, 2)
    
    @staticmethod
    def calculate_schedule(total_frames, current_window_start):
        """
        Calculate compression schedule for history frames.
        Returns a list of segments with their compression factors.
        """
        segments = []
        
        # We look backwards from current_window_start
        # Frame indices: 0 ... current_window_start-1
        
        if current_window_start <= 0:
            return []
            
        # Iterate through history frames
        # We'll group consecutive frames with the same compression factor
        
        current_segment = None
        
        for i in range(current_window_start):
            # Distance from the *start* of the generation window
            # i.e., frame (current_window_start - 1) has distance 1
            distance = current_window_start - i
            
            compression = 1
            for start, end, factor in FramePackManager.SCHEDULE_CONFIG:
                if start < distance <= end:
                    compression = factor
                    break
            
            if current_segment is None:
                current_segment = {
                    'start': i,
                    'end': i + 1,
                    'compression': compression
                }
            elif current_segment['compression'] == compression:
                current_segment['end'] = i + 1
            else:
                segments.append(current_segment)
                current_segment = {
                    'start': i,
                    'end': i + 1,
                    'compression': compression
                }
        
        if current_segment:
            segments.append(current_segment)
            
        return segments

    @staticmethod
    def pool_latents(latents, schedule, device):
        """
        Pool history latents based on the schedule.
        Input: latents tensor [B, C, T, H, W] (concatenated history)
        Output: List of compressed latent tensors
        """
        compressed_chunks = []
        
        # latents is expected to be the full history tensor
        # We need to slice it according to the schedule
        # Note: The schedule indices refer to *frames*, but latents are *latent frames*.
        # WanVideo VAE stride is (4, 8, 8).
        # So 1 latent frame = 4 pixel frames.
        # We need to map pixel frame indices to latent indices.
        
        # Actually, the input 'latents' here should probably be the *latent* representation of history.
        # If we are working with already encoded latents, we need to adjust the schedule logic 
        # or assume the schedule applies to latent frames directly.
        
        # Let's assume the schedule applies to *latent frames* for simplicity in this V1.
        # Or better, we pass the pixel-space schedule and convert.
        
        # VAE Stride T=4.
        vae_stride_t = 4
        
        for segment in schedule:
            # Convert pixel indices to latent indices
            # We take a conservative approach: include latent frames that cover the pixel range
            start_idx = segment['start'] // vae_stride_t
            end_idx = (segment['end'] + vae_stride_t - 1) // vae_stride_t
            
            # Ensure we don't go out of bounds
            if start_idx >= latents.shape[2]:
                continue
            end_idx = min(end_idx, latents.shape[2])
            
            if start_idx >= end_idx:
                continue
                
            chunk = latents[:, :, start_idx:end_idx, :, :]
            factor = segment['compression']
            
            if factor == 1:
                compressed_chunks.append(chunk)
            else:
                # Calculate pooling kernel
                # Factor 4 -> 1x2x2 pooling on latents (since latents are already 4x8x8 pixels)
                # Plan: "Mid-Term (4x): 1x4x4 effective". Base is 1x2x2.
                # So we need 1x2x2 pooling on the *patch embeddings*, not VAE latents.
                # The VAE latents are input to the model.
                # The model patches them.
                # So this function should be called inside the model forward pass, on the *embeddings*.
                
                
                # REVISION: This function will be used inside WanModel to pool *embeddings*.
                # So input 'latents' is actually 'embeddings' [B, C, T, H, W] (after patch_embedding).
                
                # Calculate kernel size for embedding pooling
                # Base patch: 1x2x2.
                # Target 4x compression -> Need to reduce token count by 4.
                # We can do 1x2x2 pooling on embeddings. (1*2*2 = 4).
                # Target 16x compression -> Need to reduce by 16.
                # We can do 2x2x4 pooling? Or 1x4x4?
                # 1x4x4 = 16.
                # Target 64x compression -> Need to reduce by 64.
                # 4x4x4 = 64.
                
                # Let's define kernel sizes for factors:
                if factor == 4:
                    kernel = (1, 2, 2)
                elif factor == 16:
                    kernel = (1, 4, 4)
                elif factor == 64:
                    kernel = (4, 4, 4)
                else:
                    kernel = (1, 1, 1)
                
                # Handle dimensions that might not be divisible
                # We use ceil_mode=True logic or padding if needed, but avg_pool3d handles boundaries reasonably.
                pooled = F.avg_pool3d(chunk, kernel_size=kernel, stride=kernel, ceil_mode=True)
                compressed_chunks.append(pooled)
                
        return compressed_chunks

    @staticmethod
    def compute_freqs_from_positions(positions, dim, theta=10000):
        """
        Generate RoPE frequencies for arbitrary 1D positions.
        Replicates logic from rope_params but for explicit positions.
        """
        # positions: [N]
        # dim: embedding dimension
        
        assert dim % 2 == 0
        # exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
        # inv_theta_pow = 1.0 / torch.pow(theta, exponents)
        
        # Use the same logic as WanVideo rope_params
        exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device).div(dim)
        inv_theta_pow = 1.0 / torch.pow(theta, exponents)
        
        freqs = torch.outer(positions.to(dtype=torch.float64), inv_theta_pow)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @staticmethod
    def generate_compressed_freqs(schedule, dim, original_grid_size, device):
        """
        Generate the full frequency tensor for the compressed history.
        original_grid_size: (F, H, W) of the *embeddings* (before pooling)
        """
        # We need to generate T, H, W frequencies separately and then combine them
        # just like rope_apply does, but for the compressed sequence.
        
        # 1. Construct the list of (t, h, w) coordinates for every token in the compressed sequence.
        
        t_freqs_list = []
        h_freqs_list = []
        w_freqs_list = []
        
        # We assume the schedule segments map to the embedding grid
        # original_grid_size is [F_emb, H_emb, W_emb]
        
        current_t = 0
        
        all_freqs = []
        
        for segment in schedule:
            factor = segment['compression']
            
            # Determine kernel size (must match pool_latents)
            if factor == 4:
                k_t, k_h, k_w = 1, 2, 2
            elif factor == 16:
                k_t, k_h, k_w = 1, 4, 4
            elif factor == 64:
                k_t, k_h, k_w = 4, 4, 4
            else:
                k_t, k_h, k_w = 1, 1, 1
            
            # Segment length in embedding frames
            # We need to know how many frames this segment covers in embedding space
            # The schedule is in *latent frames* (from calculate_schedule logic above)
            # But here we are working with *embedding frames*.
            # WanVideo: Patch Size (1, 2, 2).
            # So 1 Latent Frame = 1 Embedding Frame (Temporal patch size is 1).
            # So schedule 'start'/'end' map 1:1 to embedding temporal dimension.
            
            seg_start = segment['start']
            seg_end = segment['end']
            seg_len = seg_end - seg_start
            
            # Generate T coordinates (centroids)
            # If we pool 4 frames [0,1,2,3], centroid is 1.5
            # Local coords: linspace(start + (k-1)/2, end - (k-1)/2, steps)
            
            num_t = math.ceil(seg_len / k_t)
            t_coords = torch.linspace(
                seg_start + (k_t - 1) / 2,
                seg_end - 1 - (k_t - 1) / 2, # Approximate end
                num_t,
                device=device
            )
            
            # Generate H, W coordinates
            # H, W are spatial dimensions of the embeddings
            H_emb, W_emb = original_grid_size[1], original_grid_size[2]
            
            num_h = math.ceil(H_emb / k_h)
            h_coords = torch.linspace((k_h - 1) / 2, H_emb - 1 - (k_h - 1) / 2, num_h, device=device)
            
            num_w = math.ceil(W_emb / k_w)
            w_coords = torch.linspace((k_w - 1) / 2, W_emb - 1 - (k_w - 1) / 2, num_w, device=device)
            
            # Now we need to compute the RoPE frequencies for these coordinates
            # WanVideo splits dim into 3 parts: [d0, d1, d2] for T, H, W
            # d = dim // num_heads
            # parts = [d - 4*(d//6), 2*(d//6), 2*(d//6)]
            
            # We need the exact split logic from WanModel
            # dim passed here is usually head_dim
            
            d = dim
            d_t = d - 4 * (d // 6)
            d_h = 2 * (d // 6)
            d_w = 2 * (d // 6)
            
            freqs_t = FramePackManager.compute_freqs_from_positions(t_coords, d_t)
            freqs_h = FramePackManager.compute_freqs_from_positions(h_coords, d_h)
            freqs_w = FramePackManager.compute_freqs_from_positions(w_coords, d_w)
            
            # Now we need to broadcast and combine them into a grid for this segment
            # Shape: [num_t, num_h, num_w, dim]
            
            # freqs_t: [num_t, d_t/2] (complex)
            # We need to expand to [num_t, num_h, num_w, d_t/2]
            
            ft = freqs_t.view(num_t, 1, 1, -1).expand(num_t, num_h, num_w, -1)
            fh = freqs_h.view(1, num_h, 1, -1).expand(num_t, num_h, num_w, -1)
            fw = freqs_w.view(1, 1, num_w, -1).expand(num_t, num_h, num_w, -1)
            
            # Concatenate along last dim
            # Result: [num_t, num_h, num_w, dim/2]
            f_combined = torch.cat([ft, fh, fw], dim=-1)
            
            # Flatten to [L_segment, dim/2]
            f_flat = f_combined.reshape(-1, dim // 2)
            all_freqs.append(f_flat)
            
        # Concatenate all segments
        if not all_freqs:
            return None
            
        full_freqs = torch.cat(all_freqs, dim=0)
        return full_freqs

class FramePackCompressor:
   
    def __init__(self, 
                 lambda_compression: float = 2.0,
                 base_kernel_sizes: List[Tuple[int, int, int]] = [(2, 4, 4), (4, 8, 8), (8, 16, 16)],
                 max_history_frames: int = 100):
        
        self.lambda_compression = lambda_compression
        self.base_kernel_sizes = sorted(base_kernel_sizes, key=lambda x: x[0] * x[1] * x[2])
        self.max_history_frames = max_history_frames
        
        self.compression_schedule = self._create_compression_schedule()
    
    def _create_compression_schedule(self) -> List[Tuple[int, Tuple[int, int, int]]]:
      
        schedule = []
       
        schedule.append((1, (1, 1, 1)))
      
        for i in range(1, self.max_history_frames):
            compression_rate = int(self.lambda_compression ** i)
            kernel_size = self._get_kernel_for_compression_rate(compression_rate)
            schedule.append((compression_rate, kernel_size))
        
        return schedule
    
    def _get_kernel_for_compression_rate(self, rate: int) -> Tuple[int, int, int]:
       
        if rate == 1:
            return (1, 1, 1)
        
        for kernel in self.base_kernel_sizes:
            kernel_rate = kernel[0] * kernel[1] * kernel[2]
            if kernel_rate >= rate:
                return kernel
        
        largest_kernel = self.base_kernel_sizes[-1]
        remaining_compression = rate // (largest_kernel[0] * largest_kernel[1] * largest_kernel[2])
        
        if remaining_compression <= 8:
            return (min(remaining_compression * largest_kernel[0], 16), 
                   largest_kernel[1], largest_kernel[2])
        else:
            return (16, 32, 32)
    
    def compress_frame(self, frame: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
        
        if kernel_size == (1, 1, 1):
            return frame
        
        if frame.dim() == 4 and frame.shape[-1] < frame.shape[0]:
            frame = frame.permute(3, 0, 1, 2)
        
        # Fixed bug: removed extra frame argument
        compressed = F.avg_pool3d(
            frame.unsqueeze(0),
            kernel_size=kernel_size,
            stride=kernel_size
        ).squeeze(0) 
        
        return compressed
    
    def compress_history(self, history_frames: List[torch.Tensor]) -> List[torch.Tensor]:
        
        if not history_frames:
            return []
        
        compressed_history = []
        
        for i, frame in enumerate(history_frames):
            if i >= len(self.compression_schedule):
                kernel_size = self.base_kernel_sizes[-1]
            else:
                _, kernel_size = self.compression_schedule[i]
            
            try:
                compressed_frame = self.compress_frame(frame, kernel_size)
                compressed_history.append(compressed_frame)
            except RuntimeError as e:
                if "size" in str(e).lower():
                    compressed_frame = torch.mean(frame, dim=(1, 2, 3), keepdim=True)
                    compressed_history.append(compressed_frame)
                else:
                    raise e
        
        return compressed_history
    
    def add_new_section(self, 
                       current_history: List[torch.Tensor], 
                       new_section: torch.Tensor) -> List[torch.Tensor]:
        if len(current_history) == 0:
            return [new_section]
       
        updated_history = [new_section] + current_history
        
        compressed_history = []
        for i, section in enumerate(updated_history):
            if i >= len(self.compression_schedule):
                _, kernel_size = self.compression_schedule[-1]
            else:
                _, kernel_size = self.compression_schedule[i]
            
            compressed_section = self.compress_frame(section, kernel_size)
            compressed_history.append(compressed_section)
            
            if len(compressed_history) >= self.max_history_frames // 20:
                break
        
        return compressed_history
    
    def get_compression_stats(self, history_frames: List[torch.Tensor]) -> dict:
        
        if not history_frames:
            return {"total_frames": 0, "total_elements": 0}
        
        original_elements = sum(frame.numel() for frame in history_frames)
        compressed_history = self.compress_history(history_frames)
        compressed_elements = sum(frame.numel() for frame in compressed_history)
        
        return {
            "total_frames": len(history_frames),
            "original_elements": original_elements,
            "compressed_elements": compressed_elements,
            "compression_ratio": original_elements / compressed_elements if compressed_elements > 0 else 0,
            "memory_saved_mb": (original_elements - compressed_elements) * 4 / (1024 * 1024)  # Assuming float32
        }
        
    def select_context_frames(self, 
                        compressed_history: List[torch.Tensor], 
                        num_context_frames: int = 11,
                        add_generation_frames: bool = True) -> torch.Tensor:
        
        if not compressed_history:
            return torch.empty(0)
     
        LONG_FRAMES = 5
        MID_FRAMES = 3
        RECENT_FRAMES = 1
        OVERLAP_FRAMES = 2
        GEN_FRAMES = 30
        TOTAL_FRAMES = 41
        CONTEXT_FRAMES = 11  
        
        all_frames = []
        frame_to_section_map = [] 
        
        for section_idx, section in enumerate(compressed_history):
            num_frames_in_section = section.shape[1]
            for frame_idx in range(num_frames_in_section):
                frame = section[:, frame_idx:frame_idx+1, :, :] 
                all_frames.append(frame)
                frame_to_section_map.append(section_idx)
        
        total_available_frames = len(all_frames)
        
        if total_available_frames == 0:
            return torch.empty(0)
        
        selected_frame_indices = []
        
        if total_available_frames >= 40:
            step = max(1, (total_available_frames - 20) // LONG_FRAMES)
            for i in range(LONG_FRAMES):
                idx = i * step
                selected_frame_indices.append(idx)
        else:
            if total_available_frames >= LONG_FRAMES:
                step = total_available_frames // LONG_FRAMES
                for i in range(LONG_FRAMES):
                    selected_frame_indices.append(i * step)
            else:
                selected_frame_indices.extend(range(total_available_frames))
                while len(selected_frame_indices) < LONG_FRAMES:
                    selected_frame_indices.append(total_available_frames - 1)
       
        mid_start = max(LONG_FRAMES, total_available_frames - 15)
        for i in range(MID_FRAMES):
            idx = min(mid_start + i * 2, total_available_frames - 1)
            selected_frame_indices.append(idx)
        
        recent_idx = max(0, total_available_frames - 5)
        selected_frame_indices.append(recent_idx)
        
        for i in range(OVERLAP_FRAMES):
            idx = max(0, total_available_frames - OVERLAP_FRAMES + i)
            selected_frame_indices.append(idx)
        
        seen = set()
        unique_indices = []
        for idx in selected_frame_indices:
            if idx not in seen and idx < total_available_frames:
                unique_indices.append(idx)
                seen.add(idx)
        
        if len(unique_indices) > CONTEXT_FRAMES:
            unique_indices = unique_indices[:CONTEXT_FRAMES]
        elif len(unique_indices) < CONTEXT_FRAMES:
            while len(unique_indices) < CONTEXT_FRAMES:
                unique_indices.append(total_available_frames - 1)
        
        selected_frames = [all_frames[i] for i in unique_indices]
        
        max_h = max(f.shape[2] for f in selected_frames)
        max_w = max(f.shape[3] for f in selected_frames)
        
        padded_frames = []
        for frame in selected_frames:
            if frame.shape[2] < max_h or frame.shape[3] < max_w:
                pad_h = max_h - frame.shape[2]
                pad_w = max_w - frame.shape[3]
                frame = F.pad(frame, (0, pad_w, 0, pad_h))
            padded_frames.append(frame)
        
        context_tensor = torch.cat(padded_frames, dim=1)  
        
        if add_generation_frames:
            C, T, H, W = context_tensor.shape
            assert T == CONTEXT_FRAMES, f"Expected {CONTEXT_FRAMES} context frames, got {T}"
            
            gen_placeholder = torch.zeros((C, GEN_FRAMES, H, W), device=context_tensor.device)
            final_tensor = torch.cat([context_tensor, gen_placeholder], dim=1)
           
            assert final_tensor.shape[1] == TOTAL_FRAMES, f"Expected {TOTAL_FRAMES} frames, got {final_tensor.shape[1]}"
            
            return final_tensor
        
        return context_tensor
