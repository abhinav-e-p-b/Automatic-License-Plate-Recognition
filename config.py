"""
config.py — Central configuration for the Indian ANPR project.

All tuneable parameters live here. Import this in any script:
    from config import cfg
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    yaml_path: Path = Path("data/indian_plates.yaml")
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class ModelConfig:
    weights: str = "yolov8s.pt"         # Pretrained backbone
    best_weights: Path = Path("models/best.pt")
    img_size: int = 640
    # Detection thresholds
    conf_thresh: float = 0.50
    iou_thresh: float = 0.45


@dataclass
class TrainConfig:
    epochs: int = 50
    batch: int = 16
    lr0: float = 0.01
    lrf: float = 0.01
    patience: int = 10
    device: int = 0                     # GPU index; "cpu" for CPU
    project: str = "runs/plate_det"
    run_name: str = "v1"
    save_period: int = 10
    # Augmentation strengths (all applied via YOLOv8)
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 5.0
    perspective: float = 0.0005
    flipud: float = 0.0
    fliplr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4


@dataclass
class PreprocessConfig:
    upscale_factor: float = 2.0
    bilateral_d: int = 11
    bilateral_sigma: float = 17.0
    block_size: int = 11                # Adaptive threshold block size (odd)
    c: int = 2                          # Adaptive threshold constant
    morph_kernel: int = 2
    fix_night: bool = True              # Auto-invert dark-bg plates


@dataclass
class OCRConfig:
    gpu: bool = True
    languages: list = field(default_factory=lambda: ["en"])
    min_conf: float = 0.30


@dataclass
class VideoConfig:
    nth_frame: int = 3
    motion_thresh: float = 15.0
    cooldown_frames: int = 30


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


# Singleton — import this everywhere
cfg = Config()
