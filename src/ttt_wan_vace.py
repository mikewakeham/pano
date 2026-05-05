import gc
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from tqdm import tqdm

_third_party = Path(__file__).resolve().parent.parent / "third_party"
_wan_root = _third_party / "Wan2.1"
if str(_wan_root) not in sys.path:
    sys.path.insert(0, str(_wan_root))

from wan.configs import WAN_CONFIGS
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.vace_model import VaceWanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

VACE_FRAME_FORMULA_DIVISOR = 4


def _nearest_valid_frame_count(n: int) -> int:
    """Round down to nearest 4n+1 frame count."""
    return max(1, ((n - 1) // VACE_FRAME_FORMULA_DIVISOR) * VACE_FRAME_FORMULA_DIVISOR + 1)


def preprocess_video_and_mask(
    video_frames: np.ndarray,
    mask_frames: np.ndarray,
    target_height: int = 480,
    target_width: int = 832,
    max_frames: int = 81,
) -> list[dict]:
    """
    Resize and chunk an equirectangular video + mask into VACE-compatible batches.

    Args:
        video_frames: (N, H, W, 3) uint8 array, the rendered equirectangular video.
        mask_frames: (N, H, W) bool array where True = valid rendered pixel,
                     False = hole/missing region.
        target_height: Model target height (480 for 1.3B, 720 for 14B).
        target_width: Model target width (832 for 1.3B, 1280 for 14B).
        max_frames: Maximum frames per chunk (must be 4n+1, default 81).

    Returns:
        List of chunk dicts, each with:
          - "video": Tensor [C, F, H, W] in [-1, 1]
          - "mask": Tensor [1, F, H, W] in {0, 1} (1 = inpaint, 0 = keep)
          - "start_idx": int, original frame index of chunk start
          - "end_idx": int, original frame index of chunk end (exclusive)
    """
    assert video_frames.ndim == 4 and video_frames.shape[-1] == 3
    assert mask_frames.ndim == 3
    n_frames, orig_h, orig_w = mask_frames.shape

    max_frames = _nearest_valid_frame_count(max_frames)

    video_t = torch.from_numpy(video_frames).float().permute(0, 3, 1, 2) / 255.0
    video_t = video_t * 2.0 - 1.0  # [0,1] -> [-1,1]
    video_t = F.interpolate(video_t, size=(target_height, target_width), mode="bilinear", align_corners=False)

    # Convert mask: True (valid) -> 0 (keep), False (hole) -> 1 (inpaint)
    mask_t = torch.from_numpy(~mask_frames).float().unsqueeze(1)  # (N, 1, H, W)
    mask_t = F.interpolate(mask_t, size=(target_height, target_width), mode="nearest-exact")

    # Chunk into max_frames segments with overlap for blending
    overlap = 4 if n_frames > max_frames else 0
    chunks = []
    start = 0
    while start < n_frames:
        end = min(start + max_frames, n_frames)
        actual_len = end - start
        valid_len = _nearest_valid_frame_count(actual_len)
        if valid_len < 5 and chunks:
            # Too short for a standalone chunk; extend previous chunk
            break
        chunk_end = start + valid_len

        vid_chunk = video_t[start:chunk_end].permute(1, 0, 2, 3)  # (C, F, H, W)
        msk_chunk = mask_t[start:chunk_end].permute(1, 0, 2, 3)  # (1, F, H, W)

        chunks.append({
            "video": vid_chunk,
            "mask": msk_chunk,
            "start_idx": start,
            "end_idx": chunk_end,
        })

        if chunk_end >= n_frames:
            break
        start = chunk_end - overlap

    return chunks


# ---------------------------------------------------------------------------
# VACE encoding helpers (standalone, no class dependency)
# ---------------------------------------------------------------------------

def vace_encode_frames(
    vae: WanVAE,
    frames: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Encode video frames with mask splitting into VACE format.

    Args:
        vae: WanVAE instance.
        frames: [C, F, H, W] in [-1, 1].
        mask: [1, F, H, W] in {0, 1} (1 = inpaint region).

    Returns:
        Encoded latents with inactive+reactive concatenated: [2*z_dim, F', H', W']
    """
    mask_bin = torch.where(mask > 0.5, 1.0, 0.0)

    inactive = frames * (1 - mask_bin)
    reactive = frames * mask_bin

    inactive_latent = vae.encode([inactive])[0]  # [z_dim, F', H', W']
    reactive_latent = vae.encode([reactive])[0]  # [z_dim, F', H', W']

    return torch.cat([inactive_latent, reactive_latent], dim=0)


def vace_encode_mask(
    mask: torch.Tensor,
    vae_stride: tuple = (4, 8, 8),
) -> torch.Tensor:
    """
    Downsample mask to match VAE latent spatial resolution for VACE.

    Args:
        mask: [1, F, H, W] in {0, 1}.
        vae_stride: temporal and spatial compression factors.

    Returns:
        Encoded mask: [vae_stride[1]*vae_stride[2], F', H', W']
    """
    _, depth, height, width = mask.shape

    new_depth = int((depth + 3) // vae_stride[0])
    height = 2 * (int(height) // (vae_stride[1] * 2))
    width = 2 * (int(width) // (vae_stride[2] * 2))

    m = mask[0, :, :height * vae_stride[1], :width * vae_stride[2]]
    m = m.reshape(m.shape[0], height, vae_stride[1], width, vae_stride[2])
    m = m.permute(2, 4, 0, 1, 3)  # (stride_h, stride_w, depth, height, width)
    m = m.reshape(vae_stride[1] * vae_stride[2], m.shape[2], height, width)

    m = F.interpolate(
        m.unsqueeze(0),
        size=(new_depth, height, width),
        mode="nearest-exact",
    ).squeeze(0)

    return m


def build_vace_context(
    vae: WanVAE,
    frames: torch.Tensor,
    mask: torch.Tensor,
    vae_stride: tuple = (4, 8, 8),
) -> torch.Tensor:
    """
    Full VACE context: encoded frames + encoded mask concatenated along channels.

    Returns:
        [2*z_dim + stride_h*stride_w, F', H', W'] tensor (e.g. 32+64 = 96 channels).
    """
    z = vace_encode_frames(vae, frames, mask)
    m = vace_encode_mask(mask, vae_stride)
    return torch.cat([z, m], dim=0)


# ---------------------------------------------------------------------------
# Test-Time Training
# ---------------------------------------------------------------------------

class VaceTTT:
    """
    Test-time training for panoramic video refinement via Wan 2.1 VACE + LoRA.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: torch.device = torch.device("cuda"),
        model_size: str = "1.3B",
        lora_rank: int = 64,
        lora_alpha: int = 32,
        lora_target_modules: Optional[list[str]] = None,
        use_gradient_checkpointing: bool = True,
    ):
        self.device = device
        self.model_size = model_size
        self.checkpoint_dir = checkpoint_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q", "k", "v", "o"]
        self.use_gradient_checkpointing = use_gradient_checkpointing

        config_key = f"vace-{model_size}"
        self.config = WAN_CONFIGS[config_key]
        self.vae_stride = self.config.vae_stride
        self.patch_size = self.config.patch_size
        self.num_train_timesteps = self.config.num_train_timesteps
        self.param_dtype = self.config.param_dtype

        self._load_components()

    def _load_components(self):
        """Load VAE, T5, and VaceWanModel."""
        logger.info("Loading T5 text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
        )

        logger.info("Loading WanVAE...")
        self.vae = WanVAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
            device=self.device,
        )

        logger.info("Loading VaceWanModel...")
        self.model = VaceWanModel.from_pretrained(self.checkpoint_dir)
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

    def _encode_text(self, prompt: str) -> list[torch.Tensor]:
        """Encode a text prompt via T5."""
        self.text_encoder.model.to(self.device)
        ctx = self.text_encoder([prompt], self.device)
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        return ctx

    def _apply_lora(self):
        """Add LoRA adapters to the VaceWanModel."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights=True,
            target_modules=self.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total parameters")

    def _get_underlying_model(self):
        """Navigate PEFT wrappers to get the original VaceWanModel."""
        model = self.model
        # PeftModel -> base_model (LoraModel) -> model (VaceWanModel)
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model") and isinstance(model.model, VaceWanModel):
            model = model.model
        return model

    def _enable_gradient_checkpointing(self):
        """Wrap transformer blocks with gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint as torch_ckpt

        underlying = self._get_underlying_model()

        def _wrap_block(block):
            original_forward = block.forward
            def ckpt_forward(*args, **kwargs):
                return torch_ckpt(original_forward, *args, use_reentrant=False, **kwargs)
            block.forward = ckpt_forward

        for block in underlying.blocks:
            _wrap_block(block)

        for block in underlying.vace_blocks:
            _wrap_block(block)

    def _compute_seq_len(self, latent_shape: tuple) -> int:
        """Compute the sequence length for positional encoding."""
        _, f, h, w = latent_shape
        return math.ceil(
            (h * w) / (self.patch_size[1] * self.patch_size[2]) * f
        )

    def train(
        self,
        video_frames: np.ndarray,
        mask_frames: np.ndarray,
        prompt: str = "",
        num_steps: int = 200,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 50,
        shift: float = 5.0,
        seed: int = 42,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        save_checkpoint_path: Optional[str] = None,
    ):
        """
        Run test-time training to adapt the model to the input video.

        Args:
            video_frames: (N, H, W, 3) uint8 rendered equirectangular video.
            mask_frames: (N, H, W) bool mask (True = valid, False = hole).
            prompt: Text description of the scene.
            num_steps: Number of TTT optimization steps.
            learning_rate: LoRA learning rate.
            weight_decay: AdamW weight decay.
            max_grad_norm: Gradient clipping norm.
            warmup_steps: LR warmup steps.
            shift: Flow matching noise schedule shift.
            seed: Random seed.
            target_height: Override model target height.
            target_width: Override model target width.
            save_checkpoint_path: Optional path to save LoRA weights after training.
        """
        if target_height is None:
            target_height = 480 if self.model_size == "1.3B" else 720
        if target_width is None:
            target_width = 832 if self.model_size == "1.3B" else 1280

        torch.manual_seed(seed)

        # --- Step 1: Apply LoRA ---
        logger.info("Applying LoRA adapters...")
        self._apply_lora()

        if self.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing...")
            self._enable_gradient_checkpointing()

        # --- Step 2: Preprocess & encode data (frozen components) ---
        logger.info("Preprocessing video and mask...")
        chunks = preprocess_video_and_mask(
            video_frames, mask_frames,
            target_height=target_height,
            target_width=target_width,
            max_frames=81,
        )

        logger.info(f"Produced {len(chunks)} chunk(s) for training")

        logger.info("Encoding text prompt...")
        context = self._encode_text(prompt)
        context_null = self._encode_text("")

        logger.info("Encoding video chunks through VAE + VACE...")
        encoded_chunks = []
        for chunk in chunks:
            vid = chunk["video"].to(self.device)
            msk = chunk["mask"].to(self.device)

            vace_ctx = build_vace_context(
                self.vae, vid, msk, vae_stride=self.vae_stride
            )

            with torch.no_grad():
                target_latent = self.vae.encode([vid])[0]

            encoded_chunks.append({
                "vace_context": vace_ctx.detach(),
                "target_latent": target_latent.detach(),
            })

        # Offload VAE to CPU
        self.vae.model.cpu()
        torch.cuda.empty_cache()

        # --- Step 3: Setup optimizer ---
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        for p in trainable_params:
            p.data = p.data.float()

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # --- Step 4: Training loop ---
        logger.info(f"Starting TTT for {num_steps} steps...")
        self.model.train()

        loss_history = []
        progress_bar = tqdm(range(num_steps), desc="TTT")
        for step in progress_bar:
            chunk_idx = step % len(encoded_chunks)
            ec = encoded_chunks[chunk_idx]

            target_latent = ec["target_latent"]
            vace_ctx = ec["vace_context"]

            z_dim = target_latent.shape[0]
            seq_len = self._compute_seq_len(target_latent.shape)

            # Sample random timestep
            t_int = torch.randint(0, self.num_train_timesteps, (1,), device=self.device)
            sigma_t = t_int.float() / self.num_train_timesteps
            sigma_t = shift * sigma_t / (1.0 + (shift - 1.0) * sigma_t)

            # Flow matching interpolation: z_t = (1 - sigma) * data + sigma * noise
            noise = torch.randn_like(target_latent)
            z_t = (1.0 - sigma_t) * target_latent + sigma_t * noise

            # Flow target: v = noise - data
            flow_target = noise - target_latent

            with amp.autocast(dtype=self.param_dtype):
                flow_pred = self.model(
                    [z_t],
                    t=t_int,
                    vace_context=[vace_ctx],
                    vace_context_scale=1.0,
                    context=context,
                    seq_len=seq_len,
                )[0]

            # loss = F.mse_loss(flow_pred.float(), flow_target.float())
            weights = 1.0 / (sigma_t.clamp(min=1e-4))
            loss = (weights * (flow_pred.float() - flow_target.float()) ** 2).mean()
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        logger.info(f"TTT complete. Final loss: {loss.item():.4f}")

        # Save LoRA checkpoint and loss curve if requested
        if save_checkpoint_path:
            logger.info(f"Saving LoRA checkpoint to {save_checkpoint_path}")
            self.model.save_pretrained(save_checkpoint_path)
            loss_path = Path(save_checkpoint_path) / "loss_curve.json"
            with open(loss_path, "w") as f:
                json.dump({"step": list(range(len(loss_history))), "loss": loss_history}, f)
            logger.info(f"Saved loss curve to {loss_path}")

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        video_frames: np.ndarray,
        mask_frames: np.ndarray,
        prompt: str = "",
        n_prompt: str = "",
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        context_scale: float = 1.0,
        shift: float = 5.0,
        seed: int = 42,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run inference with the TTT-adapted model to produce refined video.

        Args:
            video_frames: (N, H, W, 3) uint8 rendered equirectangular video.
            mask_frames: (N, H, W) bool mask (True = valid, False = hole).
            prompt: Text description of the scene.
            n_prompt: Negative prompt.
            sampling_steps: Number of denoising steps.
            guide_scale: Classifier-free guidance scale.
            context_scale: VACE context injection strength.
            shift: Flow matching noise schedule shift.
            seed: Random seed.
            target_height: Override model target height.
            target_width: Override model target width.

        Returns:
            (N, H, W, 3) uint8 array of the refined equirectangular video at
            original resolution.
        """
        if target_height is None:
            target_height = 480 if self.model_size == "1.3B" else 720
        if target_width is None:
            target_width = 832 if self.model_size == "1.3B" else 1280

        orig_n, orig_h, orig_w = mask_frames.shape
        seed_g = torch.Generator(device=self.device).manual_seed(seed)

        if not n_prompt:
            n_prompt = self.config.sample_neg_prompt

        # Encode prompts
        context = self._encode_text(prompt)
        context_null = self._encode_text(n_prompt)

        # Preprocess
        chunks = preprocess_video_and_mask(
            video_frames, mask_frames,
            target_height=target_height,
            target_width=target_width,
            max_frames=81,
        )

        # Ensure VAE is on device
        self.vae.model.to(self.device)

        all_generated_frames = []
        for ci, chunk in enumerate(chunks):
            logger.info(f"Generating chunk {ci + 1}/{len(chunks)}: "
                        f"frames {chunk['start_idx']}-{chunk['end_idx']}")

            vid = chunk["video"].to(self.device)
            msk = chunk["mask"].to(self.device)

            # Build VACE context
            vace_ctx = build_vace_context(
                self.vae, vid, msk, vae_stride=self.vae_stride
            )

            z_dim = self.vae.model.z_dim
            target_shape = list(self.vae.encode([vid])[0].shape)
            noise = torch.randn(
                target_shape[0], target_shape[1],
                target_shape[2], target_shape[3],
                dtype=torch.float32, device=self.device, generator=seed_g,
            )

            seq_len = self._compute_seq_len(target_shape)

            # Setup scheduler
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latents = [noise]
            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}

            self.model.to(self.device)
            with amp.autocast(dtype=self.param_dtype):
                for t in tqdm(timesteps, desc=f"Denoising chunk {ci + 1}"):
                    ts = torch.stack([t])

                    noise_pred_cond = self.model(
                        latents, t=ts,
                        vace_context=[vace_ctx],
                        vace_context_scale=context_scale,
                        **arg_c,
                    )[0]

                    noise_pred_uncond = self.model(
                        latents, t=ts,
                        vace_context=[vace_ctx],
                        vace_context_scale=context_scale,
                        **arg_null,
                    )[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t,
                        latents[0].unsqueeze(0),
                        return_dict=False, generator=seed_g,
                    )[0]
                    latents = [temp_x0.squeeze(0)]

            # Decode
            decoded = self.vae.decode(latents)  # list of [C, F, H, W]
            vid_out = decoded[0]  # [C, F, H, W] in [-1, 1]
            all_generated_frames.append(vid_out.cpu())

            del latents, noise, vace_ctx
            torch.cuda.empty_cache()

        # Offload
        self.vae.model.cpu()
        torch.cuda.empty_cache()

        # Reassemble chunks with cross-fade blending
        output = self._reassemble_chunks(
            all_generated_frames, chunks, orig_n, orig_h, orig_w
        )
        return output

    def _reassemble_chunks(
        self,
        generated: list[torch.Tensor],
        chunks: list[dict],
        orig_n: int,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """
        Reassemble generated chunk tensors back into full-resolution video frames.

        Handles temporal blending in overlap regions and spatial upsampling.
        """
        frame_accum = torch.zeros(3, orig_n, orig_h, orig_w)
        weight_accum = torch.zeros(1, orig_n, 1, 1)

        for gen, chunk in zip(generated, chunks):
            start = chunk["start_idx"]
            end = chunk["end_idx"]
            n_chunk = end - start

            # gen is [C, F, H, W]; take only the frames that correspond to real indices
            gen_frames = gen[:, :n_chunk]

            # Upsample back to original resolution
            gen_frames_nhwc = gen_frames.permute(1, 0, 2, 3)  # (F, C, H, W)
            gen_upsampled = F.interpolate(
                gen_frames_nhwc, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )  # (F, C, H, W)
            gen_upsampled = gen_upsampled.permute(1, 0, 2, 3)  # (C, F, H, W)

            # Linear blend weights for overlap (ramp up at start, ramp down at end)
            w = torch.ones(1, n_chunk, 1, 1)
            overlap = 4
            if start > 0 and n_chunk > overlap:
                ramp = torch.linspace(0, 1, overlap).view(1, overlap, 1, 1)
                w[:, :overlap] = ramp
            if chunk["end_idx"] < orig_n and n_chunk > overlap:
                ramp = torch.linspace(1, 0, overlap).view(1, overlap, 1, 1)
                w[:, -overlap:] = ramp

            frame_accum[:, start:end] += gen_upsampled * w
            weight_accum[:, start:end] += w

        weight_accum = weight_accum.clamp(min=1e-6)
        result = frame_accum / weight_accum

        # [-1, 1] -> [0, 255]
        result = ((result + 1.0) / 2.0).clamp(0, 1) * 255.0
        result = result.permute(1, 2, 3, 0).numpy().astype(np.uint8)  # (N, H, W, 3)
        return result


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def refine_panoramic_video(
    video_frames: np.ndarray,
    mask_frames: np.ndarray,
    checkpoint_dir: str,
    prompt: str = "",
    device: torch.device = torch.device("cuda"),
    model_size: str = "1.3B",
    num_train_steps: int = 200,
    learning_rate: float = 2e-5,
    lora_rank: int = 64,
    sampling_steps: int = 50,
    guide_scale: float = 5.0,
    shift: float = 5.0,
    seed: int = 42,
    save_lora_path: Optional[str] = None,
) -> np.ndarray:
    """
    End-to-end panoramic video refinement via test-time training.

    Args:
        video_frames: (N, H, W, 3) uint8 rendered equirectangular video.
        mask_frames: (N, H, W) bool mask (True = rendered valid pixel, False = hole).
        checkpoint_dir: Path to Wan2.1-VACE model checkpoint directory.
        prompt: Text description of the scene.
        device: Torch device.
        model_size: "1.3B" or "14B".
        num_train_steps: Number of TTT optimization steps.
        learning_rate: LoRA learning rate.
        lora_rank: LoRA rank.
        sampling_steps: Number of inference denoising steps.
        guide_scale: Classifier-free guidance scale.
        shift: Noise schedule shift.
        seed: Random seed.
        save_lora_path: Optional path to save LoRA checkpoint.

    Returns:
        (N, H, W, 3) uint8 refined equirectangular video.
    """
    ttt = VaceTTT(
        checkpoint_dir=checkpoint_dir,
        device=device,
        model_size=model_size,
        lora_rank=lora_rank,
    )

    ttt.train(
        video_frames=video_frames,
        mask_frames=mask_frames,
        prompt=prompt,
        num_steps=num_train_steps,
        learning_rate=learning_rate,
        shift=shift,
        seed=seed,
        save_checkpoint_path=save_lora_path,
    )

    refined = ttt.generate(
        video_frames=video_frames,
        mask_frames=mask_frames,
        prompt=prompt,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        shift=shift,
        seed=seed,
    )

    del ttt
    gc.collect()
    torch.cuda.empty_cache()

    return refined
