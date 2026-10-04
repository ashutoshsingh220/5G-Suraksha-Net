from suraksha.config import AppConfig, load_config


def test_load_config_defaults():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.detection.conf_threshold > 0
    assert cfg.fight.temporal.window_frames >= 8
    assert cfg.api.port > 0


def test_env_override(monkeypatch):
    import suraksha.config as c

    c.load_config.cache_clear()
    monkeypatch.setenv("SURAKSHA_RTSP_URL", "rtsp://test-cam:554/live")
    monkeypatch.setenv("SURAKSHA_API_PORT", "9999")
    cfg = c.load_config()
    assert cfg.capture.rtsp_url == "rtsp://test-cam:554/live"
    assert cfg.api.port == 9999
    c.load_config.cache_clear()


def test_resolve_paths():
    cfg = load_config()
    p = cfg.resolve("configs/app.yaml")
    assert p.is_absolute() and p.exists()
