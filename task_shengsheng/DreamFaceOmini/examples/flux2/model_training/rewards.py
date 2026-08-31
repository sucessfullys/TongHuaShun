"""
Reward functions for DiffusionNFT RL training.

Provides ArcFace identity similarity, LAION aesthetic score,
and a composite weighted combination.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional


class ArcFaceReward:
    """Compute ArcFace cosine similarity between generated and reference images."""

    def __init__(self, arcface_ckpt_path: str, insightface_root: str = None,
                 device: str = "cuda", dtype=torch.bfloat16):
        from diffsynth.diffusion.loss import FaceIdentityHelper
        self.helper = FaceIdentityHelper(
            arcface_ckpt_path=arcface_ckpt_path,
            insightface_root=insightface_root,
        ).to(device=device, dtype=dtype)

    def score(self, images: List[Image.Image],
              ref_images: List[Image.Image],
              prompts: List[str] = None) -> List[float]:
        scores = []
        for gen_img, ref_img in zip(images, ref_images):
            ref_emb = self.helper.get_embedding(ref_img)
            gen_emb = self.helper.get_embedding(gen_img)
            if ref_emb is None or gen_emb is None:
                scores.append(0.0)
                continue
            cos_sim = float(np.dot(ref_emb, gen_emb) /
                            (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb) + 1e-8))
            scores.append(max(0.0, cos_sim))
        return scores


class AestheticReward:
    """LAION improved aesthetic predictor (ViT-L/14 CLIP + linear head)."""

    def __init__(self, clip_model_name: str = "ViT-L/14",
                 aesthetic_ckpt: str = None, device: str = "cuda"):
        import clip
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval().requires_grad_(False)

        if aesthetic_ckpt is not None:
            self.aesthetic_head = self._load_aesthetic_head(aesthetic_ckpt, device)
        else:
            self.aesthetic_head = None

    @staticmethod
    def _load_aesthetic_head(ckpt_path: str, device: str):
        import torch.nn as nn
        head = nn.Linear(768, 1)
        state = torch.load(ckpt_path, map_location="cpu")
        head.load_state_dict(state)
        head = head.to(device).eval()
        head.requires_grad_(False)
        return head

    @torch.no_grad()
    def score(self, images: List[Image.Image],
              ref_images: List[Image.Image] = None,
              prompts: List[str] = None) -> List[float]:
        preprocessed = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        feats = self.clip_model.encode_image(preprocessed).float()
        feats = F.normalize(feats, dim=-1)

        if self.aesthetic_head is not None:
            preds = self.aesthetic_head(feats).squeeze(-1)
            return preds.cpu().tolist()

        return feats.norm(dim=-1).cpu().tolist()


class CompositeReward:
    """Weighted combination of multiple reward functions."""

    def __init__(self, rewards: dict):
        """
        Args:
            rewards: {name: (reward_fn, weight)} dict.
        """
        self.rewards = rewards

    def score(self, images: List[Image.Image],
              ref_images: List[Image.Image] = None,
              prompts: List[str] = None) -> dict:
        all_scores = {}
        composite = [0.0] * len(images)

        for name, (reward_fn, weight) in self.rewards.items():
            scores = reward_fn.score(images, ref_images, prompts)
            all_scores[name] = scores
            for i, s in enumerate(scores):
                composite[i] += weight * s

        all_scores["composite"] = composite
        return all_scores
