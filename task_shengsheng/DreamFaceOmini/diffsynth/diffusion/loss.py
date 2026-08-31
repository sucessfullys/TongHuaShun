from .base_pipeline import BasePipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def robust_mse_loss(pred, target, threshold=50.0, spatial_weight=None):
    """
    MSE with hard outlier clipping (from FireRed-Image-Edit).
    Positions where |pred - target| > threshold are masked out,
    preventing rare extreme residuals from dominating the gradient.

    spatial_weight: optional per-token weight, shape (1, seq_len, 1),
    broadcast over batch and channels (e.g. face-region upweighting).
    """
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target, reduction='none')
    mask = ((pred - target).abs() <= threshold).float()
    if spatial_weight is not None:
        mse = mse * spatial_weight.to(device=mse.device, dtype=mse.dtype)
    return (mse * mask).mean()


# ==============================================================================
# Minimal IResNet100 definition for ArcFace (avoids insightface training dep)
# ==============================================================================

class _IResNetConv(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))

class _IResNetBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
            if stride != 1 or in_c != out_c else None
        )
    def forward(self, x):
        identity = x
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

def _make_layer(in_c, out_c, num_blocks, stride=2):
    layers = [_IResNetBlock(in_c, out_c, stride)]
    for _ in range(1, num_blocks):
        layers.append(_IResNetBlock(out_c, out_c))
    return nn.Sequential(*layers)

class IResNet100(nn.Module):
    """IResNet-100 for ArcFace: (B,3,112,112) → (B,512)."""
    def __init__(self, fp16=False):
        super().__init__()
        self.fp16 = fp16
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = _make_layer(64, 64, 3, stride=2)
        self.layer2 = _make_layer(64, 128, 13, stride=2)
        self.layer3 = _make_layer(128, 256, 30, stride=2)
        self.layer4 = _make_layer(256, 512, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512)
    def forward(self, x):
        if self.fp16:
            x = x.half()
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.features(x)
        return x


ARCFACE_DST = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)


def _compute_similarity_transform(landmark, image_size=112):
    import skimage.transform as trans
    dst = ARCFACE_DST.copy()
    if image_size % 112 == 0:
        dst *= float(image_size) / 112.0
    else:
        ratio = float(image_size) / 128.0
        dst[:, 0] += 8.0 * ratio
        dst *= ratio
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, dst)
    return tform.params[0:2, :]


def differentiable_face_crop(img_tensor, landmark_np, image_size=112):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    device = img_tensor.device
    dtype = img_tensor.dtype
    B, C, H, W = img_tensor.shape

    M = _compute_similarity_transform(landmark_np, image_size)
    M = torch.tensor(M, dtype=torch.float32, device=device)
    M_full = torch.eye(3, device=device, dtype=torch.float32)
    M_full[:2, :] = M

    src_norm = torch.tensor([
        [2.0 / W, 0, -1],
        [0, 2.0 / H, -1],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    dst_norm = torch.tensor([
        [2.0 / image_size, 0, -1],
        [0, 2.0 / image_size, -1],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)

    theta = src_norm @ torch.inverse(M_full) @ torch.inverse(dst_norm)
    theta = theta[:2, :].unsqueeze(0).expand(B, -1, -1)
    grid = F.affine_grid(theta, [B, C, image_size, image_size], align_corners=False).to(dtype=dtype)
    return F.grid_sample(img_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class FaceIdentityHelper:
    """
    Differentiable ArcFace embedding pipeline following WithAnyone (Eq. 5-6).

    Key design: use GT-image landmarks g(T) to align BOTH the generated image
    and the GT image, avoiding unreliable detection on noisy predictions.
    """

    def __init__(
        self,
        arcface_ckpt_path,
        insightface_name="buffalo_l",
        insightface_root=None,
        det_size=(640, 640),
    ):
        from insightface.app import FaceAnalysis
        self.netArc = self._load_arcface(arcface_ckpt_path)
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16

        kwargs = {"name": insightface_name}
        if insightface_root is not None:
            kwargs["root"] = insightface_root
        self.face_app = FaceAnalysis(
            **kwargs,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=-1, det_size=det_size)

    @staticmethod
    def _load_arcface(ckpt_path):
        """Load ArcFace model, handling both full-model and state_dict checkpoints."""
        loaded = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(loaded, dict):
            return loaded
        model = IResNet100(fp16=False)
        info = model.load_state_dict(loaded, strict=False)
        if info.missing_keys:
            print(f"[FaceID] Warning: missing keys when loading ArcFace: {info.missing_keys}")
        print(f"[FaceID] Loaded ArcFace state_dict into IResNet100 from {ckpt_path}")
        return model

    def to(self, device, dtype=None):
        self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        self.netArc = self.netArc.to(device=self._device, dtype=self._dtype)
        return self

    @torch.no_grad()
    def _detect_largest_face(self, img_bgr_np):
        faces = self.face_app.get(img_bgr_np)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # @staticmethod
    # def _is_profile_face(landmarks, threshold=0.6):
    #     """
    #     Detect side-facing (profile) faces via eye-distance / face-width ratio.
    #     Returns True if the face is too profile for reliable ArcFace embedding.
    #     """
    #     left_eye, right_eye = landmarks[0], landmarks[1]
    #     eye_dist = abs(right_eye[0] - left_eye[0])
    #     face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
    #     ratio = eye_dist / (face_width + 1e-6)
    #     return ratio < threshold

    @staticmethod
    def _is_profile_face(landmarks, yaw_thresh=0.25, pitch_thresh=0.5):
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        left_mouth, right_mouth = landmarks[3], landmarks[4]
        
        # Yaw (左右转): 双眼水平间距 / 脸宽
        # ratio ~1.0 正脸, ~0.5 约45°, ~0.25 约70°, ~0 背影
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]) + 1e-6
        eye_dist_x = abs(right_eye[0] - left_eye[0])
        yaw_ratio = eye_dist_x / face_width
        
        # Pitch (上下俯仰): 鼻子到眼睛中心距离 / 鼻子到嘴巴中心距离
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        nose_to_eye = abs(nose[1] - eye_center_y) + 1e-6
        nose_to_mouth = abs(mouth_center_y - nose[1]) + 1e-6
        pitch_ratio = nose_to_eye / (nose_to_eye + nose_to_mouth)
        # 正脸时 ~0.4，大幅抬头时 → 接近 0，大幅低头时 → 接近 1
        
        is_yaw_bad = yaw_ratio < yaw_thresh
        is_pitch_bad = pitch_ratio < 0.1 or pitch_ratio > 0.85
        return is_yaw_bad or is_pitch_bad

    # ------------------------------------------------------------------
    # Reference / GT image utilities  (data-loading time, no grad needed)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_embedding_and_landmarks(self, pil_image):
        """
        Detect face, extract embedding AND landmarks from a clean PIL image.
        Skips profile faces (>~45°) to avoid unreliable ArcFace embeddings.
        Returns (embedding_numpy_512, landmarks_numpy_5x2) or (None, None).
        """
        img_np = np.array(pil_image.convert("RGB"))[:, :, ::-1]
        face = self._detect_largest_face(img_np)
        if face is None:
            return None, None
        landmark = face.kps  # (5, 2) numpy

        if self._is_profile_face(landmark):
            print(f"[FaceID] Skipping profile face")
            return None, None

        img_rgb = np.array(pil_image.convert("RGB")).astype(np.float32)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        aligned = differentiable_face_crop(img_t, landmark, image_size=112)
        aligned_norm = (aligned - 127.5) / 127.5
        emb = self.netArc(aligned_norm.to(device=self._device, dtype=self._dtype))
        emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().float().numpy(), landmark.copy()

    @torch.no_grad()
    def get_embedding(self, pil_image):
        """Convenience wrapper – returns only the embedding."""
        emb, _ = self.get_embedding_and_landmarks(pil_image)
        return emb

    @staticmethod
    def embedding_to_tensor(emb):
        return torch.from_numpy(emb).unsqueeze(0).float()

    @staticmethod
    def landmarks_to_tensor(lmk):
        return torch.from_numpy(lmk).float()

    # ------------------------------------------------------------------
    # GT-Aligned differentiable embedding  (training time, grad flows)
    # ------------------------------------------------------------------

    def get_gt_aligned_embedding(self, image_tensor, gt_landmarks_np):
        """
        Extract face embedding from image_tensor using **pre-detected GT landmarks**.
        Gradients flow through image_tensor → grid_sample → ArcFace → output.

        This implements Eq. 6:  f(g(T), G)  where g(T) = gt_landmarks_np.

        Args:
            image_tensor: (B, C, H, W) in [-1, 1], gradient-carrying.
            gt_landmarks_np: (5, 2) numpy – landmarks detected from the GT image.
        Returns:
            (1, 512) normalized embedding tensor, or None on failure.
        """
        img_255 = (image_tensor + 1) * 127.5
        aligned = differentiable_face_crop(img_255, gt_landmarks_np, image_size=112)
        aligned_norm = (aligned - 127.5) / 127.5
        emb = self.netArc(aligned_norm.to(device=self._device, dtype=self._dtype))
        return F.normalize(emb, dim=-1)


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = pipe.scheduler.sample_timestep_id(min_timestep_boundary, max_timestep_boundary)
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    loss_spatial_weight = inputs.pop("loss_spatial_weight", None)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = robust_mse_loss(noise_pred, training_target, spatial_weight=loss_spatial_weight)
    return loss


def _info_nce_loss(anchor, positive, negatives, temperature=0.07):
    """
    InfoNCE contrastive loss (Eq. 7).

    Args:
        anchor:    (D,) generated-image embedding.
        positive:  (D,) same-identity reference embedding.
        negatives: (M, D) different-identity embeddings.
        temperature: scalar τ.
    Returns:
        scalar loss.
    """
    pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0)) / temperature
    neg_sims = F.cosine_similarity(anchor.unsqueeze(0), negatives, dim=-1) / temperature
    logits = torch.cat([pos_sim, neg_sims], dim=0)
    labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits.unsqueeze(0), labels)


def FlowMatchSFTLossWithFaceID(
    pipe: BasePipeline,
    face_helper: FaceIdentityHelper,
    face_id_weight: float = 0.1,
    face_cl_weight: float = 0.1,
    face_cl_temperature: float = 0.07,
    **inputs,
):
    """
    L = L_diff + λ_ID · L_ID + λ_CL · L_CL   (Eq. 8)

    L_ID (Eq. 6):  GT-aligned cosine loss across all noise levels.
        Uses GT-image landmarks g(T) to align both generated G and GT T.
    L_CL (Eq. 7):  InfoNCE contrastive loss with extended negatives.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = pipe.scheduler.sample_timestep_id(min_timestep_boundary, max_timestep_boundary)
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    target_for_mse = training_target
    pred_for_mse = noise_pred
    if "first_frame_latents" in inputs:
        pred_for_mse = noise_pred[:, :, 1:]
        target_for_mse = training_target[:, :, 1:]

    loss_diff = robust_mse_loss(pred_for_mse, target_for_mse)

    gt_face_landmarks = inputs.get("gt_face_landmarks", None)
    gt_face_emb = inputs.get("gt_face_emb", None)

    if gt_face_landmarks is None or gt_face_emb is None:
        return loss_diff

    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma_weight = (1.0 - sigma).clamp(min=0.0)

    z_t = inputs["latents"]
    x0_hat = z_t - sigma * noise_pred

    H = int(inputs["height"]) // 16
    W = int(inputs["width"]) // 16
    x0_hat_spatial = rearrange(x0_hat, "B (H W) C -> B C H W", H=H, W=W)

    with torch.no_grad():
        pipe.load_models_to_device(["vae"])
    decoded = pipe.vae.decode(x0_hat_spatial.to(pipe.torch_dtype))

    gt_lmk_np = gt_face_landmarks
    if isinstance(gt_lmk_np, torch.Tensor):
        gt_lmk_np = gt_lmk_np.cpu().float().numpy()

    gen_emb = face_helper.get_gt_aligned_embedding(decoded, gt_lmk_np)

    if gen_emb is None:
        return loss_diff

    gt_emb = gt_face_emb.to(device=gen_emb.device, dtype=gen_emb.dtype)
    gt_emb = F.normalize(gt_emb, dim=-1)
    cos_sim = (gen_emb * gt_emb).sum(dim=-1)
    loss_id = (1.0 - cos_sim).mean()

    loss = loss_diff + face_id_weight * sigma_weight * loss_id

    loss_cl = torch.tensor(0.0, device=loss.device)
    negative_pool = inputs.get("negative_face_pool", None)
    if face_cl_weight > 0 and negative_pool is not None and negative_pool.shape[0] > 0:
        neg = negative_pool.to(device=gen_emb.device, dtype=gen_emb.dtype)
        neg = F.normalize(neg, dim=-1)
        ref_emb = gt_emb.squeeze(0)
        loss_cl = _info_nce_loss(
            gen_emb.squeeze(0), ref_emb, neg,
            temperature=face_cl_temperature,
        )
        loss = loss + face_cl_weight * sigma_weight * loss_cl

    return {
        "loss": loss,
        "loss_diff": loss_diff.detach(),
        "loss_id": loss_id.detach(),
        "loss_cl": loss_cl.detach(),
        "face_cos_sim": cos_sim.mean().detach(),
        "sigma": sigma.detach(),
    }


def FlowMatchSFTLossFireRedID(
    pipe: BasePipeline,
    face_helper: FaceIdentityHelper,
    face_id_weight: float = 0.1,
    sigma_cap: float = 0.9,
    **inputs,
):
    """
    FireRed-style identity-preserving loss:

        L = L_diff + η · σ² · L_ID

    Key differences from FlowMatchSFTLossWithFaceID:
    - σ² weighting: stronger identity supervision at high noise where
      the model's x̂₀ drifts to wrong identities.
    - No contrastive (InfoNCE) loss — identity is supervised directly.
    - ArcFace forward in float32 for gradient precision.

    Differentiability chain (following WithAnyone):
        noise_pred → x̂₀ → VAE.decode → face_crop → ArcFace → cos_sim → loss
        Gradients flow all the way back to the DiT through this chain.
        VAE and ArcFace are frozen (no param updates) but NOT wrapped
        in no_grad, so backward pass reaches the DiT.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = pipe.scheduler.sample_timestep_id(min_timestep_boundary, max_timestep_boundary)
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    target_for_mse = training_target
    pred_for_mse = noise_pred
    if "first_frame_latents" in inputs:
        pred_for_mse = noise_pred[:, :, 1:]
        target_for_mse = training_target[:, :, 1:]

    loss_diff = robust_mse_loss(pred_for_mse, target_for_mse)

    # --- Face identity loss with FireRed σ² weighting ---
    gt_face_landmarks = inputs.get("gt_face_landmarks", None)
    gt_face_emb = inputs.get("gt_face_emb", None)

    if gt_face_landmarks is None or gt_face_emb is None:
        return loss_diff

    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma_weight = (sigma ** 2).clamp(max=sigma_cap ** 2)

    # x̂₀ = x_t − σ · v̂  (gradient flows through noise_pred)
    z_t = inputs["latents"]
    x0_hat = z_t - sigma * noise_pred

    H = int(inputs["height"]) // 16
    W = int(inputs["width"]) // 16
    x0_hat_spatial = rearrange(x0_hat, "B (H W) C -> B C H W", H=H, W=W)

    # Load VAE without blocking gradients (only the model-loading is no_grad)
    with torch.no_grad():
        pipe.load_models_to_device(["vae"])

    # VAE decode: gradients flow through (VAE params frozen but not no_grad)
    decoded = pipe.vae.decode(x0_hat_spatial.to(pipe.torch_dtype))

    gt_lmk_np = gt_face_landmarks
    if isinstance(gt_lmk_np, torch.Tensor):
        gt_lmk_np = gt_lmk_np.cpu().float().numpy()

    # Scale to [0, 255] for face crop; ArcFace forward in model dtype (bf16),
    # then cast to float32 for precise normalization (following WithAnyone)
    img_255 = (decoded + 1) * 127.5
    aligned = differentiable_face_crop(img_255, gt_lmk_np, image_size=112)
    aligned_norm = (aligned - 127.5) / 127.5
    gen_emb = face_helper.netArc(aligned_norm.to(device=face_helper._device, dtype=face_helper._dtype))
    gen_emb = F.normalize(gen_emb.float(), dim=-1)

    if gen_emb is None:
        return loss_diff

    # L_ID = 1 − cos(ArcFace(x̂₀), ArcFace(GT))
    gt_emb = gt_face_emb.to(device=gen_emb.device, dtype=torch.float32)
    gt_emb = F.normalize(gt_emb, dim=-1)
    cos_sim = (gen_emb * gt_emb).sum(dim=-1)
    loss_id = (1.0 - cos_sim).mean()

    loss = loss_diff + face_id_weight * sigma_weight * loss_id

    return {
        "loss": loss,
        "loss_diff": loss_diff.detach(),
        "loss_id": loss_id.detach(),
        "loss_cl": torch.tensor(0.0),
        "face_cos_sim": cos_sim.mean().detach(),
        "sigma": sigma.detach(),
        "sigma_weight": sigma_weight.detach(),
    }


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def FlowMatchNFTLoss(
    pipe: BasePipeline,
    model_fn_old,
    model_fn_ref,
    advantage: float = 0.0,
    nft_beta: float = 0.5,
    kl_beta: float = 0.01,
    adv_clip_max: float = 5.0,
    **inputs,
):
    """
    DiffusionNFT loss: reward-weighted forward-process RL for flow matching.

    Instead of standard SFT (reconstruct GT), this uses reward advantages to
    weight between positive (reward-aligned) and negative (reward-opposed)
    velocity branches, plus KL regularization to the frozen base model.

    Reference: Zheng et al., "DiffusionNFT: Online Diffusion Reinforcement
    with Forward Process", ICLR 2026 Oral.

    Args:
        pipe: pipeline with scheduler and model_fn (current trainable policy).
        model_fn_old: velocity prediction from old/EMA policy (no grad).
        model_fn_ref: velocity prediction from base model without LoRA (no grad).
        advantage: scalar advantage for this sample (from per-prompt z-score).
        nft_beta: interpolation coefficient for positive/negative velocity mixing.
        kl_beta: weight for KL regularization to base model.
        adv_clip_max: advantage clipping range.
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = pipe.scheduler.sample_timestep_id(min_timestep_boundary, max_timestep_boundary)
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    sigma = pipe.scheduler.sigmas[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    x0 = inputs["input_latents"]
    x_t = pipe.scheduler.add_noise(x0, noise, timestep)
    inputs["latents"] = x_t

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    v_current = pipe.model_fn(**models, **inputs, timestep=timestep)

    with torch.no_grad():
        v_old = model_fn_old(**inputs, timestep=timestep)
        v_ref = model_fn_ref(**inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        v_current = v_current[:, :, 1:]
        v_old = v_old[:, :, 1:]
        v_ref = v_ref[:, :, 1:]
        x_t = x_t[:, :, 1:]
        x0 = x0[:, :, 1:]

    beta = nft_beta
    v_pos = beta * v_current + (1 - beta) * v_old
    v_neg = (1 + beta) * v_old - beta * v_current

    sigma_expanded = sigma
    while sigma_expanded.dim() < v_pos.dim():
        sigma_expanded = sigma_expanded.unsqueeze(-1)

    x0_pos = x_t - sigma_expanded * v_pos
    x0_neg = x_t - sigma_expanded * v_neg

    weight_factor = (x0_pos.float() - x0.float()).abs().mean().clamp(min=1e-5)
    pos_loss = F.mse_loss(x0_pos.float(), x0.float(), reduction='none').mean() / weight_factor
    neg_loss = F.mse_loss(x0_neg.float(), x0.float(), reduction='none').mean() / weight_factor

    adv_clipped = max(min(advantage, adv_clip_max), -adv_clip_max)
    r = (adv_clipped / adv_clip_max / 2.0 + 0.5)
    r = max(0.0, min(1.0, r))

    policy_loss = adv_clip_max * (r / beta * pos_loss + (1 - r) / beta * neg_loss)

    kl_loss = F.mse_loss(v_current.float(), v_ref.float())
    loss = policy_loss + kl_beta * kl_loss

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "pos_loss": pos_loss.detach(),
        "neg_loss": neg_loss.detach(),
        "advantage": torch.tensor(advantage),
        "r_weight": torch.tensor(r),
        "sigma": sigma.detach(),
    }


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
