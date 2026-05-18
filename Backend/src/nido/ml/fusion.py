def _margen_audio(audio_probs: dict[str, float]) -> float:
    """Diferencia top1-top2. Alta = audio muy seguro."""
    probs = sorted(audio_probs.values(), reverse=True)
    if len(probs) < 2:
        return 1.0
    return float(probs[0] - probs[1])


def fusion_filtro_umbral(
    audio_probs: dict[str, float],
    geo_probs: dict[str, float],
    umbral_bajo: float = 0.002,
    umbral_medio: float = 0.01,
    factor_bajo: float = 0.05,
    factor_medio: float = 0.3,
    margen_certeza_audio: float = 0.5,
    top_k: int = 5,
) -> tuple[list[dict], bool]:
    """
    Fusiona predicciones de audio y geo usando filtro por umbral.

    Si el audio es muy seguro (margen top1-top2 >= margen_certeza_audio)
    → usa solo audio sin filtrar.

    Si el audio es dudoso
    → penaliza especies con baja probabilidad geoespacial.

    Retorna (top_k_predicciones, filtro_aplicado)
    """
    filtro_aplicado = False

    if _margen_audio(audio_probs) >= margen_certeza_audio:
        # Audio muy seguro, usar directamente
        scores = audio_probs.copy()
    else:
        # Aplicar filtro geoespacial
        filtro_aplicado = True
        scores = {}
        for especie, audio_p in audio_probs.items():
            geo_p = geo_probs.get(especie, 0.0)
            if geo_p < umbral_bajo:
                penalizacion = factor_bajo
            elif geo_p < umbral_medio:
                penalizacion = factor_medio
            else:
                penalizacion = 1.0
            scores[especie] = audio_p * penalizacion

    # Normalizar
    total = sum(scores.values())
    if total > 0:
        scores = {e: p / total for e, p in scores.items()}

    # Ordenar y retornar top_k
    ranking = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    predictions = [
        {"scientific_name": especie, "confidence": round(prob, 4)}
        for especie, prob in ranking
    ]

    return predictions, filtro_aplicado
