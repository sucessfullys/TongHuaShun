import torch, os, argparse, accelerate
import numpy as np
from collections import deque
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    DataProcessingPipeline, RouteByType, SequencialProcess,
    ToAbsolutePath, LoadImage, ImageCropAndResize,
    RealWorldDegradation, GeometricAugmentation, HeadCropAugmentation,
)
from diffsynth.core.data.face_bbox_utils import (
    build_latent_face_weight,
    collect_bbox_entries,
    transform_bbox_norm_to_processed,
)
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, Flux2Unit_EditImageEmbedder, ModelConfig
from diffsynth.diffusion import *
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Flux2ImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        face_id_weight=0.0,
        face_cl_weight=0.0,
        face_cl_temperature=0.07,
        negative_pool_size=256,
        arcface_ckpt_path=None,
        insightface_root=None,
        face_id_mode="withanyone",
        face_sigma_cap=0.9,
        face_mse_weight=1.0,
        face_mask_expand=1.4,
        face_bbox_min_match_score=0.0,
        face_bbox_fallback=True,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"))
        self.pipe = Flux2ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.face_id_weight = face_id_weight
        self.face_cl_weight = face_cl_weight
        self.face_cl_temperature = face_cl_temperature
        self.face_id_mode = face_id_mode
        self.face_sigma_cap = face_sigma_cap
        self.face_helper = None
        self._negative_pool = deque(maxlen=negative_pool_size)

        # Face-region weighted MSE (detection only, no ArcFace / no ID loss)
        self.face_mse_weight = face_mse_weight
        self.face_mask_expand = face_mask_expand
        self.face_bbox_min_match_score = face_bbox_min_match_score
        self.face_bbox_fallback = face_bbox_fallback
        self.face_mask_app = None
        if face_mse_weight > 1.0 and face_bbox_fallback:
            from insightface.app import FaceAnalysis
            fa_kwargs = {"name": "buffalo_l", "allowed_modules": ["detection"]}
            if insightface_root is not None:
                fa_kwargs["root"] = insightface_root
            old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                self.face_mask_app = FaceAnalysis(**fa_kwargs, providers=["CPUExecutionProvider"])
                self.face_mask_app.prepare(ctx_id=-1, det_size=(640, 640))
            finally:
                if old_cuda is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
                else:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

        if face_id_weight > 0 or face_cl_weight > 0:
            if arcface_ckpt_path is None:
                raise ValueError("--arcface_ckpt_path is required when face losses are enabled")
            from diffsynth.diffusion.loss import FaceIdentityHelper
            self.face_helper = FaceIdentityHelper(
                arcface_ckpt_path=arcface_ckpt_path,
                insightface_root=insightface_root,
            ).to(device=device, dtype=torch.bfloat16)

        def _sft_loss(pipe, inputs_shared, inputs_posi, inputs_nega):
            if self.face_helper is not None:
                if self.face_id_mode == "firered":
                    return FlowMatchSFTLossFireRedID(
                        pipe,
                        face_helper=self.face_helper,
                        face_id_weight=self.face_id_weight,
                        sigma_cap=self.face_sigma_cap,
                        **inputs_shared, **inputs_posi,
                    )
                return FlowMatchSFTLossWithFaceID(
                    pipe,
                    face_helper=self.face_helper,
                    face_id_weight=self.face_id_weight,
                    face_cl_weight=self.face_cl_weight,
                    face_cl_temperature=self.face_cl_temperature,
                    **inputs_shared, **inputs_posi,
                )
            return FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi)

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": _sft_loss,
            "sft:train": _sft_loss,
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def build_face_loss_weight_from_data(self, data):
        """Prefer precomputed gt_face_bboxes; fall back to online InsightFace detection."""
        if self.face_mse_weight <= 1.0:
            return None

        processed = data["image"]
        proc_w, proc_h = processed.size

        gt_face_bboxes = data.get("gt_face_bboxes")
        if gt_face_bboxes:
            bbox_norms = collect_bbox_entries(
                gt_face_bboxes,
                prompt=data.get("prompt"),
                min_match_score=self.face_bbox_min_match_score,
                only_prompt_refs=True,
            )
            if bbox_norms:
                gt_size = data.get("gt_image_size")
                if gt_size and len(gt_size) == 2:
                    orig_w, orig_h = int(gt_size[0]), int(gt_size[1])
                else:
                    orig_w, orig_h = proc_w, proc_h
                pixels = [
                    transform_bbox_norm_to_processed(b, orig_w, orig_h, proc_w, proc_h)
                    for b in bbox_norms
                ]
                return build_latent_face_weight(
                    pixels,
                    proc_w,
                    proc_h,
                    face_mse_weight=self.face_mse_weight,
                    face_mask_expand=self.face_mask_expand,
                )

        if self.face_mask_app is None:
            return None
        return self.build_face_loss_weight_online(processed)

    def build_face_loss_weight_online(self, image):
        """
        Build per-token loss weights at latent resolution (image/16 grid).
        Face bboxes (expanded by face_mask_expand) get face_mse_weight,
        the rest 1.0; normalized to mean 1 to keep the loss scale stable.
        Returns (1, h16*w16, 1) tensor, or None if no face is detected.
        """
        faces = self.face_mask_app.get(np.array(image)[:, :, ::-1])
        if not faces:
            return None
        w16, h16 = image.size[0] // 16, image.size[1] // 16
        weight = torch.ones((h16, w16), dtype=torch.float32)
        for face in faces:
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half_w = (x2 - x1) / 2 * self.face_mask_expand
            half_h = (y2 - y1) / 2 * self.face_mask_expand
            tx1 = max(int((cx - half_w) / 16), 0)
            ty1 = max(int((cy - half_h) / 16), 0)
            tx2 = min(int((cx + half_w) / 16) + 1, w16)
            ty2 = min(int((cy + half_h) / 16) + 1, h16)
            if tx2 > tx1 and ty2 > ty1:
                weight[ty1:ty2, tx1:tx2] = self.face_mse_weight
        weight = weight / weight.mean()
        # Latents are flattened row-major: "B C H W -> B (H W) C"
        return weight.reshape(1, -1, 1)

    def get_pipeline_inputs(self, data):
        prompt = data["prompt"]
        if random.random() < 0.1:
            prompt = ""
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        if self.face_mse_weight > 1.0:
            loss_weight = self.build_face_loss_weight_from_data(data)
            if loss_weight is not None:
                inputs_shared["loss_spatial_weight"] = loss_weight

        if self.face_helper is not None:
            gt_image = data["image"]
            gt_emb, gt_lmk = self.face_helper.get_embedding_and_landmarks(gt_image)

            if gt_emb is not None and gt_lmk is not None:
                inputs_shared["gt_face_emb"] = self.face_helper.embedding_to_tensor(gt_emb)
                inputs_shared["gt_face_landmarks"] = self.face_helper.landmarks_to_tensor(gt_lmk)

                if self.face_id_mode != "firered" and self.face_cl_weight > 0 and len(self._negative_pool) > 0:
                    neg_stack = torch.stack(list(self._negative_pool), dim=0)
                    inputs_shared["negative_face_pool"] = neg_stack

                if self.face_id_mode != "firered":
                    self._negative_pool.append(
                        torch.from_numpy(gt_emb).float()
                    )

        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        result = self.task_to_loss[self.task](self.pipe, *inputs)
        if isinstance(result, dict):
            self._last_metrics = {k: v for k, v in result.items() if k != "loss"}
            return result["loss"]
        self._last_metrics = {}
        return result


def flux2_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--degradation_prob", type=float, default=0.0, help="Probability of applying real-world degradation to edit_image. 0.0 disables degradation.")
    parser.add_argument("--face_id_weight", type=float, default=0.0, help="Weight λ_ID for GT-aligned ArcFace identity loss (paper: 0.1).")
    parser.add_argument("--face_cl_weight", type=float, default=0.0, help="Weight λ_CL for InfoNCE contrastive loss (paper: 0.1).")
    parser.add_argument("--face_cl_temperature", type=float, default=0.07, help="Temperature τ for InfoNCE contrastive loss.")
    parser.add_argument("--negative_pool_size", type=int, default=256, help="Max size of the online negative embedding queue for InfoNCE.")
    parser.add_argument("--arcface_ckpt_path", type=str, default=None, help="Path to PyTorch ArcFace checkpoint (e.g., glintr100.pth).")
    parser.add_argument("--insightface_root", type=str, default=None, help="InsightFace model root path for detector weights.")
    parser.add_argument("--head_crop_prob", type=float, default=0.0, help="Probability of cropping edit_image to the head region during training. 0.0 disables head crop.")
    parser.add_argument("--geo_aug_prob", type=float, default=0.0, help="Probability of geometric augmentation (flip+rotation) on edit_image. 0.0 disables.")
    parser.add_argument("--geo_aug_max_rotation", type=float, default=15.0, help="Max rotation angle in degrees for geometric augmentation.")
    parser.add_argument("--face_id_mode", type=str, default="withanyone", choices=["withanyone", "firered"],
                        help="ID loss mode: 'withanyone' = (1-σ) + InfoNCE, 'firered' = σ² weighting, no contrastive loss.")
    parser.add_argument("--face_sigma_cap", type=float, default=0.9, help="σ cap for FireRed σ² weighting (loss zeroed above this).")
    parser.add_argument("--face_mse_weight", type=float, default=1.0, help="MSE loss multiplier for face regions of the target image (detection only, no ArcFace). 1.0 disables.")
    parser.add_argument("--face_mask_expand", type=float, default=1.4, help="Expansion ratio of detected face bboxes for the loss weight mask (covers hair/head).")
    parser.add_argument("--face_bbox_min_match_score", type=float, default=0.0, help="Min match_score in gt_face_bboxes to apply face MSE weight.")
    parser.add_argument("--no_face_bbox_fallback", action="store_true", help="Do not run online InsightFace when gt_face_bboxes is missing.")
    return parser


if __name__ == "__main__":
    parser = flux2_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # Build edit_image augmentation chain: geo_aug → head_crop → degradation → load+resize
    edit_image_augmentations = []
    if args.geo_aug_prob > 0:
        edit_image_augmentations.append(GeometricAugmentation(
            probability=args.geo_aug_prob,
            max_rotation_deg=args.geo_aug_max_rotation,
        ))
    if args.head_crop_prob > 0:
        from insightface.app import FaceAnalysis
        fa_kwargs = {"name": "buffalo_l"}
        if args.insightface_root is not None:
            fa_kwargs["root"] = args.insightface_root
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            face_app = FaceAnalysis(**fa_kwargs, providers=["CPUExecutionProvider"])
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
        finally:
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        edit_image_augmentations.append(HeadCropAugmentation(face_app=face_app, probability=args.head_crop_prob))
    if args.degradation_prob > 0:
        edit_image_augmentations.append(RealWorldDegradation(probability=args.degradation_prob))

    base_image_op = UnifiedDataset.default_image_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
    )

    edit_image_operator = None
    if edit_image_augmentations:
        load_op = ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(
            args.height, args.width, args.max_pixels, 16, 16,
        )
        aug_chain = DataProcessingPipeline(edit_image_augmentations)
        single_op = load_op >> aug_chain
        edit_image_operator = RouteByType(operator_map=[
            (str, single_op),
            (list, SequencialProcess(single_op)),
        ])

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=base_image_op,
        edit_image_operator=edit_image_operator,
    )
    model = Flux2ImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        face_id_weight=args.face_id_weight,
        face_cl_weight=args.face_cl_weight,
        face_cl_temperature=args.face_cl_temperature,
        negative_pool_size=args.negative_pool_size,
        arcface_ckpt_path=args.arcface_ckpt_path,
        insightface_root=args.insightface_root,
        face_id_mode=args.face_id_mode,
        face_sigma_cap=args.face_sigma_cap,
        face_mse_weight=args.face_mse_weight,
        face_mask_expand=args.face_mask_expand,
        face_bbox_min_match_score=args.face_bbox_min_match_score,
        face_bbox_fallback=not args.no_face_bbox_fallback,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
