import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "schedule_config.json"

_timer: threading.Timer | None = None

DEFAULT_CONFIG = {
    "enabled": True,
    "frequency": "weekly",  # daily, weekly, monthly
    "min_samples": 10,
    "next_run": None,
}

FREQUENCY_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def _save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)


def _compute_next_run(frequency: str) -> str:
    days = FREQUENCY_DAYS.get(frequency, 7)
    next_r = datetime.now().replace(hour=3, minute=0, second=0) + timedelta(days=days)
    return next_r.isoformat()


def get_schedule_status() -> dict:
    config = _load_config()
    next_run = config.get("next_run")

    # Si no hay next_run calculado, calcularlo ahora
    if not next_run:
        next_run = _compute_next_run(config.get("frequency", "weekly"))
        config["next_run"] = next_run
        _save_config(config)

    return {
        "enabled": config.get("enabled", True),
        "frequency": config.get("frequency", "weekly"),
        "min_samples": config.get("min_samples", 10),
        "next_run": next_run,
    }


def update_schedule_config(
    enabled: bool,
    frequency: str,
    min_samples: int,
) -> dict:
    config = _load_config()
    config["enabled"] = enabled
    config["frequency"] = frequency
    config["min_samples"] = min_samples
    config["next_run"] = _compute_next_run(frequency)
    _save_config(config)

    # Reiniciar el timer si está habilitado
    if enabled:
        _start_scheduler()
    else:
        _stop_scheduler()

    return get_schedule_status()


def _run_scheduled_pipeline():
    """Corre el pipeline y reprograma el siguiente."""
    config = _load_config()
    if not config.get("enabled", True):
        return

    print(f"[Scheduler] Ejecutando pipeline automático — {datetime.now().isoformat()}")

    try:
        import subprocess
        import sys

        pipeline_path = Path(__file__).resolve().parent / "flow.py"
        subprocess.run(
            [sys.executable, str(pipeline_path)],
            check=True,
            capture_output=True,
        )
    except Exception as e:
        print(f"[Scheduler] Error en pipeline automático: {e}")
    finally:
        # Actualizar next_run y reprogramar
        config["next_run"] = _compute_next_run(config.get("frequency", "weekly"))
        _save_config(config)
        _start_scheduler()


def _start_scheduler():
    global _timer
    _stop_scheduler()

    config = _load_config()
    next_run = config.get("next_run")
    if not next_run or not config.get("enabled", True):
        return

    next_dt = datetime.fromisoformat(next_run)
    delay = (next_dt - datetime.now()).total_seconds()

    if delay <= 0:
        # Ya pasó la hora programada, ejecutar en 1 minuto
        delay = 60

    _timer = threading.Timer(delay, _run_scheduled_pipeline)
    _timer.daemon = True
    _timer.start()
    print(f"[Scheduler] Próximo reentrenamiento en {round(delay/3600, 1)} horas")


def _stop_scheduler():
    global _timer
    if _timer:
        _timer.cancel()
        _timer = None
