import os
import tempfile
from datetime import datetime
from typing import Optional

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

_analyzer: Optional[Analyzer] = None


def get_analyzer() -> Analyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer()
    return _analyzer


def analyze_audio(
    audio_bytes: bytes,
    latitude: float,
    longitude: float,
    recorded_at: datetime,
    elevation: Optional[int] = None,
    min_confidence: float = 0.1,
) -> list[dict]:
    analyzer = get_analyzer()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        recording = Recording(
            analyzer,
            tmp_path,
            lat=latitude,
            lon=longitude,
            date=recorded_at,
            min_conf=min_confidence,
        )
        recording.analyze()

        species_scores: dict[str, list[float]] = {}
        for detection in recording.detections:
            name = detection["scientific_name"]
            confidence = detection["confidence"]
            if name not in species_scores:
                species_scores[name] = []
            species_scores[name].append(confidence)

        results = []
        for scientific_name, scores in species_scores.items():
            original = next(
                d
                for d in recording.detections
                if d["scientific_name"] == scientific_name
            )
            results.append(
                {
                    "scientific_name": scientific_name,
                    "common_name_en": original.get("common_name", ""),
                    "confidence": round(sum(scores) / len(scores), 4),
                    "n_detections": len(scores),
                }
            )

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:5]

    except Exception:
        return []

    finally:
        os.unlink(tmp_path)
