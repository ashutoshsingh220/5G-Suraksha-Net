"""Typed configuration: YAML files + .env overrides via pydantic-settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_serializer, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CaptureConfig(BaseModel):
    rtsp_url: str = "rtsp://localhost:554/stream1"
    source_type: str | None = None  # None (auto) | file | webcam | rtsp | stream
    camera_index: int = 0  # device index for local webcam (0 = default integrated/USB camera)
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 15
    reconnect_delay_s: float = 3.0
    max_reconnect_attempts: int = -1
    # throttle = cap at target_fps | realtime = pace at source fps (file/webcam
    # simulation) | fast = no pacing, decode+process as fast as possible (benchmark)
    pace: str = "throttle"
    max_frames: int = 0  # 0 = unlimited; >0 stops after N emitted frames


class DetectionConfig(BaseModel):
    weights: str = "models/yolo11s.pt"
    person_class_id: int = 0
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    imgsz: int = 640


class WeaponConfirmationConfig(BaseModel):
    confirm_conf_threshold: float = 0.75
    min_hits: int = 3
    window_size: int = 5
    iou_match_threshold: float = 0.25
    expiry_frames: int = 15


class WeaponConfig(BaseModel):
    enabled: bool = True
    weights: str = "outputs/weapon_training/best.pt"
    conf_threshold: float = 0.70
    iou_threshold: float = 0.45
    inference_interval: int = 1
    device: str = "auto"
    imgsz: int = 640
    confirmation: WeaponConfirmationConfig = Field(default_factory=WeaponConfirmationConfig)


class TrackingConfig(BaseModel):
    tracker_config: str = "configs/bytetrack.yaml"
    persist: bool = True


class MovementConfig(BaseModel):
    flow_grid_size: int = 32
    panic_speed_threshold: float = 2.2
    panic_min_tracks: int = 3


class CrowdConfig(BaseModel):
    zones_file: str = "configs/zones.yaml"
    density_high_threshold: float = 0.35
    density_critical_threshold: float = 0.55
    growth_window_s: int = 30
    growth_alert_per_min: float = 10.0
    enable_density_incidents: bool = False  # Density is context/telemetry; does not trigger incidents alone
    min_panic_persons: int = 8  # Minimum headcount in scene before crowd panic triggers incident
    movement: MovementConfig = Field(default_factory=MovementConfig)


class CandidateConfig(BaseModel):
    min_pair_speed: float = 1.6
    proximity_iou: float = 0.08
    # Center distance fallback in body-widths when bounding boxes do not overlap.
    # Calibrated for leaning over / grappling postures: 1.25 body widths.
    proximity_distance: float = 1.25
    proximity_min_frames: int = 8  # ~0.5s at 15 FPS persistence requirement
    missed_frames_tolerance: int = 2  # Grace frames for ByteTrack jitter during close grappling
    # Normalized motion gate in BODY-WIDTHS/SEC (feature[2]).
    # Calibrated: 0.12 body-widths/sec separates calm talking/standing (<0.08) from active physical interaction (>=0.15).
    motion_energy_threshold: float = 0.12
    motion_variance_threshold: float = 0.0  # Candidate generator detects active interaction, GRU evaluates struggle
    decay_timeout_s: float = 1.0  # Time without motion before candidate decays to NORMAL
    cooldown_s: float = 2.0


class TemporalConfig(BaseModel):
    window_frames: int = 32
    stride_frames: int = 8
    model_weights: str = ""
    score_threshold: float = 0.65


class VerifyConfig(BaseModel):
    min_duration_s: float = 2.5
    min_consecutive_windows: int = 3
    min_motion_variance: float = 0.0
    incident_cooldown_s: float = 60.0


class FightConfig(BaseModel):
    candidate: CandidateConfig = Field(default_factory=CandidateConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)


class IncidentsConfig(BaseModel):
    snapshot_dir: str = "outputs/snapshots"
    clip_dir: str = "outputs/clips"
    report_dir: str = "outputs/incidents"
    clip_seconds_before: int = 5
    clip_seconds_after: int = 5
    clip_fps: int = 15
    async_write: bool = False


class FusionConfig(BaseModel):
    enabled: bool = True
    correlation_window_s: float = 2.5
    spatial_proximity_threshold: float = 1.5  # body widths
    escalation_enabled: bool = True
    weapon_cooldown_s: float = 30.0
    armed_fight_cooldown_s: float = 60.0


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100


class LocationConfig(BaseModel):
    mode: str = "demo"  # demo | gps | camera | unknown
    query_from_env: bool = True
    demo_query: str | None = None
    google_maps_api_key: str | None = Field(default=None, repr=False)

    def get_effective_query(self) -> str:
        if self.query_from_env:
            q = (
                os.environ.get("DEMO_LOCATION_QUERY")
                or os.environ.get("SURAKSHA_DEMO_LOCATION_QUERY")
                or self.demo_query
                or ""
            )
            return q.strip()
        return (self.demo_query or "").strip()

    def get_api_key(self) -> str | None:
        key = (
            os.environ.get("GOOGLE_MAPS_API_KEY")
            or os.environ.get("SURAKSHA_GOOGLE_MAPS_API_KEY")
            or self.google_maps_api_key
        )
        return key.strip() if key else None

    def __repr__(self) -> str:
        key_repr = "'***REDACTED***'" if (self.google_maps_api_key or os.environ.get("GOOGLE_MAPS_API_KEY")) else "None"
        return f"LocationConfig(mode={self.mode!r}, query_from_env={self.query_from_env!r}, demo_query={self.demo_query!r}, google_maps_api_key={key_repr})"

    @field_serializer("google_maps_api_key", when_used="always")
    def _mask_google_maps_api_key(self, v: str | None) -> str | None:
        return "***REDACTED***" if v else None

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        if "google_maps_api_key" in d and d["google_maps_api_key"]:
            d["google_maps_api_key"] = "***REDACTED***"
        return d


class EmailConfig(BaseModel):
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str | None = Field(default=None, repr=False)
    email_from: str = ""
    email_to: list[str] = Field(default_factory=list)          # fallback / general recipients
    email_to_police: list[str] = Field(default_factory=list)   # police station contact email(s)
    email_to_hospital: list[str] = Field(default_factory=list) # trauma center / hospital email(s)
    use_tls: bool = True
    max_attachment_mb: float = 25.0
    timeout_s: float = 60.0

    @field_validator("email_to", "email_to_police", "email_to_hospital", mode="before")
    @classmethod
    def _parse_email_to(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            if not v.strip():
                return []
            return [email.strip() for email in v.split(",") if email.strip()]
        if isinstance(v, (list, tuple)):
            return [str(e).strip() for e in v if str(e).strip()]
        return []

    def get_police_recipients(self) -> list[str]:
        """Return police-specific recipients, falling back to general email_to."""
        if self.email_to_police:
            return list(self.email_to_police)
        return list(self.email_to)

    def get_hospital_recipients(self) -> list[str]:
        """Return hospital/trauma-center-specific recipients, falling back to general email_to."""
        if self.email_to_hospital:
            return list(self.email_to_hospital)
        return list(self.email_to)

    def get_password(self) -> str | None:
        if self.smtp_password is not None and self.smtp_password != "":
            return self.smtp_password
        return (
            os.environ.get("SMTP_PASSWORD")
            or os.environ.get("SURAKSHA_SMTP_PASSWORD")
        )

    def __repr__(self) -> str:
        pw_repr = "'***REDACTED***'" if (self.smtp_password or os.environ.get("SMTP_PASSWORD") or os.environ.get("SURAKSHA_SMTP_PASSWORD")) else "None"
        return (
            f"EmailConfig(enabled={self.enabled!r}, smtp_host={self.smtp_host!r}, "
            f"smtp_port={self.smtp_port!r}, smtp_username={self.smtp_username!r}, "
            f"smtp_password={pw_repr}, email_from={self.email_from!r}, "
            f"email_to={self.email_to!r}, email_to_police={self.email_to_police!r}, "
            f"email_to_hospital={self.email_to_hospital!r}, use_tls={self.use_tls!r}, "
            f"max_attachment_mb={self.max_attachment_mb!r}, timeout_s={self.timeout_s!r})"
        )

    @field_serializer("smtp_password", when_used="always")
    def _mask_smtp_password(self, v: str | None) -> str | None:
        return "***REDACTED***" if v else None

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        d = super().model_dump(*args, **kwargs)
        if "smtp_password" in d and d["smtp_password"]:
            d["smtp_password"] = "***REDACTED***"
        return d


class TelemetryConfig(BaseModel):
    enabled: bool = True
    connection_string: str = "udpin:127.0.0.1:14550"
    baud_rate: int = 57600
    timeout_s: float = 3.0
    broadcast_rate_hz: float = 5.0


class AppConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    weapon: WeaponConfig = Field(default_factory=WeaponConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    crowd: CrowdConfig = Field(default_factory=CrowdConfig)
    fight: FightConfig = Field(default_factory=FightConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    incidents: IncidentsConfig = Field(default_factory=IncidentsConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    device: str = "auto"
    log_level: str = "INFO"
    camera_id: str = "cam_default"

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a possibly-relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env(data: dict[str, Any]) -> None:
    """Map SURAKSHA_* env vars onto the config dict."""
    env_map = {
        "SURAKSHA_RTSP_URL": ("capture", "rtsp_url"),
        "SURAKSHA_SOURCE_TYPE": ("capture", "source_type"),
        "SURAKSHA_CAMERA_INDEX": ("capture", "camera_index"),
        "SURAKSHA_PACE": ("capture", "pace"),
        "SURAKSHA_TARGET_FPS": ("capture", "target_fps"),
        "SURAKSHA_DEVICE": ("device",),
        "SURAKSHA_YOLO_WEIGHTS": ("detection", "weights"),
        "SURAKSHA_WEAPON_ENABLED": ("weapon", "enabled"),
        "SURAKSHA_WEAPON_WEIGHTS": ("weapon", "weights"),
        "SURAKSHA_WEAPON_CONF": ("weapon", "conf_threshold"),
        "SURAKSHA_WEAPON_INTERVAL": ("weapon", "inference_interval"),
        "SURAKSHA_FIGHT_MODEL_PATH": ("fight", "temporal", "model_weights"),
        "SURAKSHA_FUSION_ENABLED": ("fusion", "enabled"),
        "SURAKSHA_INCIDENTS_ASYNC": ("incidents", "async_write"),
        "SURAKSHA_API_HOST": ("api", "host"),
        "SURAKSHA_API_PORT": ("api", "port"),
        "SURAKSHA_LOG_LEVEL": ("log_level",),
        "SURAKSHA_CAMERA_ID": ("camera_id",),
        "DEMO_LOCATION_QUERY": ("location", "demo_query"),
        "SURAKSHA_DEMO_LOCATION_QUERY": ("location", "demo_query"),
        "GOOGLE_MAPS_API_KEY": ("location", "google_maps_api_key"),
        "SURAKSHA_GOOGLE_MAPS_API_KEY": ("location", "google_maps_api_key"),
        "SURAKSHA_LOCATION_MODE": ("location", "mode"),
        "SURAKSHA_LOCATION_QUERY_FROM_ENV": ("location", "query_from_env"),
        "EMAIL_ENABLED": ("email", "enabled"),
        "SURAKSHA_EMAIL_ENABLED": ("email", "enabled"),
        "SMTP_HOST": ("email", "smtp_host"),
        "SURAKSHA_SMTP_HOST": ("email", "smtp_host"),
        "SMTP_PORT": ("email", "smtp_port"),
        "SURAKSHA_SMTP_PORT": ("email", "smtp_port"),
        "SMTP_USERNAME": ("email", "smtp_username"),
        "SURAKSHA_SMTP_USERNAME": ("email", "smtp_username"),
        "SMTP_PASSWORD": ("email", "smtp_password"),
        "SURAKSHA_SMTP_PASSWORD": ("email", "smtp_password"),
        "EMAIL_FROM": ("email", "email_from"),
        "SURAKSHA_EMAIL_FROM": ("email", "email_from"),
        "EMAIL_TO": ("email", "email_to"),
        "SURAKSHA_EMAIL_TO": ("email", "email_to"),
        "EMAIL_TO_POLICE": ("email", "email_to_police"),
        "SURAKSHA_EMAIL_TO_POLICE": ("email", "email_to_police"),
        "EMAIL_TO_HOSPITAL": ("email", "email_to_hospital"),
        "SURAKSHA_EMAIL_TO_HOSPITAL": ("email", "email_to_hospital"),
        "EMAIL_USE_TLS": ("email", "use_tls"),
        "SURAKSHA_EMAIL_USE_TLS": ("email", "use_tls"),
        "EMAIL_MAX_ATTACHMENT_MB": ("email", "max_attachment_mb"),
        "SURAKSHA_EMAIL_MAX_ATTACHMENT_MB": ("email", "max_attachment_mb"),
        "SURAKSHA_TELEMETRY_ENABLED": ("telemetry", "enabled"),
        "SURAKSHA_TELEMETRY_URL": ("telemetry", "connection_string"),
        "SURAKSHA_MAVLINK_URL": ("telemetry", "connection_string"),
        "SURAKSHA_TELEMETRY_TIMEOUT": ("telemetry", "timeout_s"),
        "SURAKSHA_TELEMETRY_RATE_HZ": ("telemetry", "broadcast_rate_hz"),
    }
    for env_key, path in env_map.items():
        val = os.environ.get(env_key)
        if val is None or val == "":
            continue
        node: Any = data
        for key in path[:-1]:
            node = node.setdefault(key, {})
        leaf = path[-1]
        if leaf in ("port", "camera_index", "inference_interval", "smtp_port", "baud_rate"):
            val = int(val)  # type: ignore[assignment]
        elif leaf in ("conf_threshold", "correlation_window_s", "spatial_proximity_threshold", "max_attachment_mb", "timeout_s", "broadcast_rate_hz"):
            val = float(val)  # type: ignore[assignment]
        elif leaf in ("enabled", "async_write", "escalation_enabled", "query_from_env", "use_tls"):
            val = str(val).lower() in ("true", "1", "yes")  # type: ignore[assignment]
        node[leaf] = val


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configs/app.yaml, overlay .env, return typed AppConfig."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "app.yaml"
    data = _load_yaml(path)
    _apply_env(data)
    cfg = AppConfig(**data)
    cfg.detection.weights = str(cfg.resolve(cfg.detection.weights))
    if cfg.weapon.weights:
        cfg.weapon.weights = str(cfg.resolve(cfg.weapon.weights))
    return cfg


# ---------------------------------------------------------------------------
# Training configuration (configs/training.yaml)
#
# Kept separate from AppConfig on purpose: AppConfig is the runtime contract
# shared with the orchestration module, and adding training knobs to it would
# widen that surface for no benefit. The only coupling that matters is that the
# window geometry matches fight.temporal — enforced by TrainingConfig.validate
# against the live config rather than left to convention.
# ---------------------------------------------------------------------------

class TrainingSampleConfig(BaseModel):
    survey_videos_per_cell: int = 20
    smoke_windows: int = 64
    overfit_videos_per_cell: int = 8
    overfit_val_videos_per_cell: int = 2


class ExtractionConfig(BaseModel):
    frame_width: int = 640
    frame_height: int = 360
    device: str = "auto"
    max_frames_per_clip: int = 0


class TrainingDatasetConfig(BaseModel):
    manifest: str = "rwf2000_v1"
    labels: list[str] = Field(default_factory=lambda: ["NON_FIGHT", "FIGHT"])
    train_split: str = "train"
    val_split: str = "val"
    window_frames: int = 32
    stride_frames: int = 8
    feature_dim: int = 8
    feature_cache_dir: str = "datasets/processed/features"
    sample: TrainingSampleConfig = Field(default_factory=TrainingSampleConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)


class NormalizationConfig(BaseModel):
    method: str = "standard"
    eps: float = 1e-6


class AugmentationConfig(BaseModel):
    enabled: bool = False
    reason: str = ""


class TrainingModelConfig(BaseModel):
    feature_dim: int = 8
    hidden_size: int = 64
    head_hidden: int = 32
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.0


class OptimizerConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 64
    eval_batch_size: int = 256
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    balanced_class_weights: bool = True
    early_stopping_patience: int = 0
    seed: int = 42
    deterministic_cuda: bool = False
    device: str = "auto"
    num_workers: int = 0


class EvaluationConfig(BaseModel):
    threshold: float = 0.5
    report_leakage_adjusted: bool = True


class LeakageConfig(BaseModel):
    registry_path: str = "datasets/reports/rwf2000_duplicate_leakage.json"


class TrainingOutputsConfig(BaseModel):
    checkpoint_dir: str = "models/temporal"
    report_dir: str = "datasets/reports"
    survey_report: str = "rwf2000_feature_yield.json"
    verification_report: str = "rwf2000_pipeline_verification.json"


class BenchmarkConfig(BaseModel):
    batch_sizes: list[int] = Field(default_factory=lambda: [8, 16, 32, 64])
    iters: int = 30
    warmup: int = 5
    corpus_sizes: list[int] = Field(default_factory=lambda: [5000, 20725])


class TrainingConfig(BaseModel):
    dataset: TrainingDatasetConfig = Field(default_factory=TrainingDatasetConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: TrainingModelConfig = Field(default_factory=TrainingModelConfig)
    training: OptimizerConfig = Field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    outputs: TrainingOutputsConfig = Field(default_factory=TrainingOutputsConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    def resolve(self, relative: str | Path) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def validate_against_app(self, app: "AppConfig") -> list[str]:
        """Return inconsistencies with the live fight config (empty = OK).

        A checkpoint trained with different window geometry than
        fight.temporal loads fine but scores the wrong thing at inference, so
        this is checked rather than assumed.
        """
        problems: list[str] = []
        t = app.fight.temporal
        if t.window_frames != self.dataset.window_frames:
            problems.append(
                f"window_frames mismatch: training={self.dataset.window_frames} "
                f"app.fight.temporal={t.window_frames}"
            )
        if t.stride_frames != self.dataset.stride_frames:
            problems.append(
                f"stride_frames mismatch: training={self.dataset.stride_frames} "
                f"app.fight.temporal={t.stride_frames}"
            )
        from suraksha.training.features import FEATURE_DIM

        if self.dataset.feature_dim != FEATURE_DIM:
            problems.append(
                f"feature_dim {self.dataset.feature_dim} != "
                f"FightCandidateDetector.FEATURE_DIM {FEATURE_DIM}"
            )
        if self.model.feature_dim != self.dataset.feature_dim:
            problems.append(
                f"model.feature_dim {self.model.feature_dim} != "
                f"dataset.feature_dim {self.dataset.feature_dim}"
            )
        from suraksha.training.model import (
            BIDIRECTIONAL,
            DROPOUT,
            HEAD_HIDDEN,
            NUM_LAYERS,
        )

        for name, got, want in (
            ("head_hidden", self.model.head_hidden, HEAD_HIDDEN),
            ("num_layers", self.model.num_layers, NUM_LAYERS),
            ("bidirectional", self.model.bidirectional, BIDIRECTIONAL),
            ("dropout", self.model.dropout, DROPOUT),
        ):
            if got != want:
                problems.append(
                    f"model.{name}={got!r} differs from the shipped inference "
                    f"network ({want!r}); the checkpoint would not match"
                )
        return problems


@lru_cache(maxsize=1)
def load_training_config(config_path: str | Path | None = None) -> TrainingConfig:
    """Load configs/training.yaml as a typed TrainingConfig."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "training.yaml"
    data = _load_yaml(path)
    env_model = os.environ.get("SURAKSHA_FIGHT_MODEL_PATH")
    cfg = TrainingConfig(**data)
    if env_model:
        # keeps the checkpoint location overridable without editing YAML
        cfg.outputs.checkpoint_dir = str(Path(env_model).parent)
    return cfg


class ImcPiConfig(BaseModel):
    host: str = "10.254.18.48"
    user: str = "student"
    ssh_port: int = 22
    mediamtx_path: str = "/home/student/mediamtx"
    camera_device: str = "/dev/video0"


class ImcRtspConfig(BaseModel):
    port: int = 8554
    path: str = "drone"

    @property
    def stream_path(self) -> str:
        p = self.path.lstrip("/")
        return f"/{p}"


class ImcCameraConfig(BaseModel):
    width: int = 640
    height: int = 360
    fps: int = 15
    gop_size: int = 15
    input_format: str = "mjpeg"
    encoder: str = "libx264"
    preset: str = "ultrafast"
    tune: str = "zerolatency"
    pix_fmt: str = "yuv420p"


class ImcProbeConfig(BaseModel):
    timeout_s: float = 20.0
    max_retries: int = 5
    retry_interval_s: float = 1.0


class ImcConfig(BaseModel):
    pi: ImcPiConfig = Field(default_factory=ImcPiConfig)
    rtsp: ImcRtspConfig = Field(default_factory=ImcRtspConfig)
    camera: ImcCameraConfig = Field(default_factory=ImcCameraConfig)
    probe: ImcProbeConfig = Field(default_factory=ImcProbeConfig)

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://{self.pi.host}:{self.rtsp.port}/{self.rtsp.path.lstrip('/')}"

    @property
    def ffmpeg_command(self) -> str:
        """Command executed on the Pi to stream camera to local MediaMTX."""
        return (
            f"ffmpeg -f v4l2 -input_format {self.camera.input_format} "
            f"-video_size {self.camera.width}x{self.camera.height} "
            f"-framerate {self.camera.fps} -i {self.pi.camera_device} "
            f"-c:v {self.camera.encoder} -preset {self.camera.preset} "
            f"-tune {self.camera.tune} -g {self.camera.gop_size} -pix_fmt {self.camera.pix_fmt} "
            f"-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:{self.rtsp.port}/{self.rtsp.path.lstrip('/')}"
        )


def load_imc_config(config_path: str | Path | None = None) -> ImcConfig:
    """Load configs/imc_demo.yaml with environment-variable overrides."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "imc_demo.yaml"
    data = _load_yaml(path) if path.exists() else {}
    cfg = ImcConfig(**data)

    if "SURAKSHA_IMC_PI_HOST" in os.environ:
        cfg.pi.host = os.environ["SURAKSHA_IMC_PI_HOST"]
    if "SURAKSHA_IMC_PI_USER" in os.environ:
        cfg.pi.user = os.environ["SURAKSHA_IMC_PI_USER"]
    if "SURAKSHA_IMC_MEDIAMTX_PATH" in os.environ:
        cfg.pi.mediamtx_path = os.environ["SURAKSHA_IMC_MEDIAMTX_PATH"]
    if "SURAKSHA_IMC_CAMERA_DEVICE" in os.environ:
        cfg.pi.camera_device = os.environ["SURAKSHA_IMC_CAMERA_DEVICE"]
    if "SURAKSHA_IMC_RTSP_PORT" in os.environ:
        cfg.rtsp.port = int(os.environ["SURAKSHA_IMC_RTSP_PORT"])
    if "SURAKSHA_IMC_RTSP_PATH" in os.environ:
        cfg.rtsp.path = os.environ["SURAKSHA_IMC_RTSP_PATH"]
    if "SURAKSHA_IMC_CAMERA_WIDTH" in os.environ:
        cfg.camera.width = int(os.environ["SURAKSHA_IMC_CAMERA_WIDTH"])
    if "SURAKSHA_IMC_CAMERA_HEIGHT" in os.environ:
        cfg.camera.height = int(os.environ["SURAKSHA_IMC_CAMERA_HEIGHT"])
    if "SURAKSHA_IMC_CAMERA_FPS" in os.environ:
        cfg.camera.fps = int(os.environ["SURAKSHA_IMC_CAMERA_FPS"])
    return cfg

