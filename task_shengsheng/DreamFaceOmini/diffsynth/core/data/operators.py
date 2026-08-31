import math, random, io
import numpy as np
import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image, ImageFilter
import torchaudio


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class RealWorldDegradation(DataProcessingOperator):
    """
    Real-ESRGAN style two-stage degradation pipeline for simulating real-world image corruption.
    Each stage: gaussian blur -> random resize -> gaussian noise -> JPEG compression.

    Supports curriculum learning via warmup: probability ramps from `prob_start` to
    `probability` over `warmup_steps` training steps. Call `step()` once per training
    iteration to advance the schedule.

    Reference: https://github.com/xinntao/Real-ESRGAN
    """
    INTERPOLATION_MODES = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

    def __init__(
        self,
        probability=0.7,
        prob_start=0.0,
        warmup_steps=0,
        prob_schedule="linear",
        blur_prob=0.5, blur_sigma_range=(0.1, 2.0),
        resize_prob=0.1, resize_scale_range=(0.3, 1.0),
        noise_prob=0.5, noise_sigma_range=(1.0, 15.0), gray_noise_prob=0.4,
        jpeg_prob=0.5, jpeg_quality_range=(50, 95),
        second_degradation_prob=0.3,
        blur_sigma_range2=(0.1, 1.0),
        resize_scale_range2=(0.5, 1.0),
        noise_sigma_range2=(1.0, 10.0),
        jpeg_quality_range2=(50, 95),
    ):
        self.probability = probability
        self.prob_start = prob_start
        self.warmup_steps = warmup_steps
        self.prob_schedule = prob_schedule
        self.current_step = 0
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.resize_prob = resize_prob
        self.resize_scale_range = resize_scale_range
        self.noise_prob = noise_prob
        self.noise_sigma_range = noise_sigma_range
        self.gray_noise_prob = gray_noise_prob
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.second_degradation_prob = second_degradation_prob
        self.blur_sigma_range2 = blur_sigma_range2
        self.resize_scale_range2 = resize_scale_range2
        self.noise_sigma_range2 = noise_sigma_range2
        self.jpeg_quality_range2 = jpeg_quality_range2

    def get_effective_probability(self):
        if self.warmup_steps <= 0:
            return self.probability
        progress = min(1.0, self.current_step / self.warmup_steps)
        if self.prob_schedule == "cosine":
            return self.prob_start + (self.probability - self.prob_start) * (1 - math.cos(progress * math.pi)) / 2
        return self.prob_start + (self.probability - self.prob_start) * progress

    def step(self):
        self.current_step += 1

    def _gaussian_blur(self, image, sigma_range):
        sigma = random.uniform(*sigma_range)
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _random_resize(self, image, scale_range):
        w, h = image.size
        scale = random.uniform(*scale_range)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        down_interp = random.choice(self.INTERPOLATION_MODES)
        up_interp = random.choice(self.INTERPOLATION_MODES)
        image = image.resize((new_w, new_h), down_interp)
        image = image.resize((w, h), up_interp)
        return image

    def _gaussian_noise(self, image, sigma_range, gray_noise_prob):
        img = np.array(image, dtype=np.float32)
        sigma = random.uniform(*sigma_range)
        if random.random() < gray_noise_prob:
            noise = np.random.normal(0, sigma, img.shape[:2])
            noise = np.stack([noise] * img.shape[2], axis=-1)
        else:
            noise = np.random.normal(0, sigma, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _jpeg_compression(self, image, quality_range):
        quality = random.randint(int(quality_range[0]), int(quality_range[1]))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _single_degradation(self, image, blur_sigma, resize_scale, noise_sigma, jpeg_quality):
        if random.random() < self.blur_prob:
            image = self._gaussian_blur(image, blur_sigma)
        if random.random() < self.resize_prob:
            image = self._random_resize(image, resize_scale)
        if random.random() < self.noise_prob:
            image = self._gaussian_noise(image, noise_sigma, self.gray_noise_prob)
        if random.random() < self.jpeg_prob:
            image = self._jpeg_compression(image, jpeg_quality)
        return image

    def __call__(self, data: Image.Image):
        if random.random() > self.get_effective_probability():
            return data
        image = data.copy()
        image = self._single_degradation(
            image, self.blur_sigma_range, self.resize_scale_range,
            self.noise_sigma_range, self.jpeg_quality_range,
        )
        if random.random() < self.second_degradation_prob:
            image = self._single_degradation(
                image, self.blur_sigma_range2, self.resize_scale_range2,
                self.noise_sigma_range2, self.jpeg_quality_range2,
            )
        return image


class GeometricAugmentation(DataProcessingOperator):
    """
    Geometric augmentation on edit_image to prevent copy-paste shortcuts.

    Applies random horizontal flip and small-angle rotation so the model
    cannot simply copy pixel layout from the reference image.
    Rotation uses edge-pixel fill to avoid large black/white borders.
    """

    def __init__(self, probability=0.5, flip_prob=0.5, max_rotation_deg=15.0):
        self.probability = probability
        self.flip_prob = flip_prob
        self.max_rotation_deg = max_rotation_deg

    def __call__(self, data: Image.Image):
        if random.random() > self.probability:
            return data
        image = data
        op = random.choice(["flip", "rotate", "both", "none"])
        if op in ("flip", "both"):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if op in ("rotate", "both"):
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=None)
        return image


class HeadCropAugmentation(DataProcessingOperator):
    """
    Head-crop augmentation for face identity training.

    Detects the largest face in the image via a pre-built InsightFace FaceAnalysis
    instance, expands the bounding box by configurable margins (larger upward to
    include forehead/hair), and returns the cropped region at its native size.
    Downstream pipeline (edit_image_auto_resize) handles rescaling.

    Falls back to the original image when no face is detected or the crop is
    too small. Triggers with fixed probability on each call.
    """

    def __init__(
        self,
        face_app,
        probability=0.5,
        side_margin_ratio=0.8,
        top_margin_ratio=0.6,
        bottom_margin_ratio=1.0,
    ):
        self.face_app = face_app
        self.probability = probability
        self.side_margin_ratio = side_margin_ratio
        self.top_margin_ratio = top_margin_ratio
        self.bottom_margin_ratio = bottom_margin_ratio

    def _detect_largest_face(self, img_bgr):
        faces = self.face_app.get(img_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def __call__(self, data: Image.Image):
        if random.random() > self.probability:
            return data
        w, h = data.size
        max_det_side = 640
        if max(w, h) > max_det_side:
            scale = max_det_side / max(w, h)
            det_img = data.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        else:
            scale = 1.0
            det_img = data
        img_bgr = np.array(det_img.convert("RGB"))[:, :, ::-1]
        face = self._detect_largest_face(img_bgr)
        if face is None:
            return data
        x1, y1, x2, y2 = face.bbox[0] / scale, face.bbox[1] / scale, face.bbox[2] / scale, face.bbox[3] / scale
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, x1 - bw * self.side_margin_ratio)
        cy1 = max(0, y1 - bh * self.top_margin_ratio)
        cx2 = min(w, x2 + bw * self.side_margin_ratio)
        cy2 = min(h, y2 + bh * self.bottom_margin_ratio)
        cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return data
        return data.crop((cx1, cy1, cx2, cy2))


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_rate=24, fix_frame_rate=False):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate

    def get_reader(self, data: str):
        return imageio.get_reader(data)

    def get_available_num_frames(self, reader):
        if not self.fix_frame_rate:
            return reader.count_frames()
        meta_data = reader.get_meta_data()
        total_original_frames = int(reader.count_frames())
        duration = meta_data["duration"] if "duration" in meta_data else total_original_frames / meta_data['fps']
        total_available_frames = math.floor(duration * self.frame_rate)
        return int(total_available_frames)

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        total_frames = self.get_available_num_frames(reader)
        if int(total_frames) < num_frames:
            num_frames = total_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int) -> int:
        if not self.fix_frame_rate:
            return new_sequence_id
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))        
        frame_id = min(frame_id, total_raw_frames - 1)
        return frame_id


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, frame_rate=24, fix_frame_rate=False):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        reader = self.get_reader(data)
        raw_frame_rate = reader.get_meta_data()['fps']
        num_frames = self.get_num_frames(reader)
        total_raw_frames = reader.count_frames()
        frames = []
        for frame_id in range(num_frames):
            frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, total_raw_frames)
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)

    def __call__(self, data: str):
        reader = self.get_reader(data)
        num_frames = self.get_num_frames(reader)
        duration = num_frames / self.frame_rate
        waveform, sample_rate = torchaudio.load(data)
        target_samples = int(duration * sample_rate)
        current_samples = waveform.shape[-1]
        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform, sample_rate
