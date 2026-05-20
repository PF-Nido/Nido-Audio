from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nido.serving.db.database import get_db

router = APIRouter(prefix="/species", tags=["Especies"])


@router.get("/heatmap")
async def get_heatmap(
    species: Optional[str] = Query(None, description="Filtrar por especie científica"),
    family: Optional[str] = Query(None, description="Filtrar por familia"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Filtrar por mes"),
    elevation_min: Optional[int] = Query(0, ge=0),
    elevation_max: Optional[int] = Query(5500, le=6000),
    precision: int = Query(
        4, ge=1, le=8, description="Precisión del grid (más alto = más detalle)"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna puntos agregados para el mapa de calor.
    Agrupa observaciones en una grilla para no sobrecargar el frontend.
    precision controla el tamaño de cada celda del grid.
    """

    # Construir filtros dinámicos
    filters = [
        "r.location IS NOT NULL",
        "r.elevation BETWEEN :elevation_min AND :elevation_max",
    ]
    params: dict = {
        "elevation_min": elevation_min,
        "elevation_max": elevation_max,
        "precision": precision,
    }

    if species:
        filters.append("s.scientific_name ILIKE :species")
        params["species"] = f"%{species}%"

    if family:
        filters.append("f.scientific_name ILIKE :family")
        params["family"] = f"%{family}%"

    if month:
        filters.append("EXTRACT(MONTH FROM r.recorded_at) = :month")
        params["month"] = month

    where_clause = " AND ".join(filters)

    query = text(
        f"""
        SELECT
            ROUND(ST_Y(r.location::geometry)::numeric, :precision) AS lat,
            ROUND(ST_X(r.location::geometry)::numeric, :precision) AS lon,
            COUNT(*) AS n_observations,
            COUNT(DISTINCT r.species_id) AS n_species,
            AVG(r.elevation) AS avg_elevation
        FROM recordings r
        JOIN species s ON r.species_id = s.id
        JOIN families f ON s.family_id = f.id
        WHERE {where_clause}
        GROUP BY
            ROUND(ST_Y(r.location::geometry)::numeric, :precision),
            ROUND(ST_X(r.location::geometry)::numeric, :precision)
        ORDER BY n_observations DESC
        LIMIT 5000
    """
    )

    result = await db.execute(query, params)
    rows = result.fetchall()

    return {
        "type": "heatmap",
        "total_points": len(rows),
        "data": [
            {
                "lat": float(row.lat),
                "lon": float(row.lon),
                "n_observations": int(row.n_observations),
                "n_species": int(row.n_species),
                "avg_elevation": (
                    round(float(row.avg_elevation), 1) if row.avg_elevation else None
                ),
            }
            for row in rows
        ],
    }


@router.get("/by-location")
async def get_species_by_location(
    lat: float = Query(..., ge=-4.5, le=13.0),
    lon: float = Query(..., ge=-82.0, le=-66.0),
    radius_km: float = Query(50, ge=1, le=500),
    species: Optional[str] = Query(None),
    family: Optional[str] = Query(None),
    month: Optional[int] = Query(None, ge=1, le=12),
    elevation_min: int = Query(0, ge=0),
    elevation_max: int = Query(5500, le=6000),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna especies observadas dentro de un radio alrededor de un punto.
    Se usa cuando el usuario hace zoom o click en el mapa.
    """

    filters = [
        "r.location IS NOT NULL",
        "r.elevation BETWEEN :elevation_min AND :elevation_max",
    ]
    params: dict = {
        "lat": lat,
        "lon": lon,
        "radius_m": radius_km * 1000,
        "elevation_min": elevation_min,
        "elevation_max": elevation_max,
        "limit": limit,
    }

    if species:
        filters.append("s.scientific_name ILIKE :species")
        params["species"] = f"%{species}%"

    if family:
        filters.append("f.scientific_name ILIKE :family")
        params["family"] = f"%{family}%"

    if month:
        filters.append("EXTRACT(MONTH FROM r.recorded_at) = :month")
        params["month"] = month

    where_clause = " AND ".join(filters)

    query = text(
        f"""
        SELECT
            s.scientific_name,
            s.common_name_es,
            s.common_name_en,
            f.scientific_name AS family,
            s.iucn_status,
            s.is_endemic,
            s.is_migratory,
            COUNT(r.id) AS n_observations,
            AVG(r.elevation) AS avg_elevation,
            MIN(r.elevation) AS min_elevation,
            MAX(r.elevation) AS max_elevation,
            ST_Y(ST_Centroid(ST_Collect(r.location::geometry))) AS centroid_lat,
            ST_X(ST_Centroid(ST_Collect(r.location::geometry))) AS centroid_lon
        FROM recordings r
        JOIN species s ON r.species_id = s.id
        JOIN families f ON s.family_id = f.id
        WHERE {where_clause}
        GROUP BY
            s.id, s.scientific_name, s.common_name_es,
            s.common_name_en, f.scientific_name,
            s.iucn_status, s.is_endemic, s.is_migratory
        ORDER BY n_observations DESC
        LIMIT :limit
    """
    )

    result = await db.execute(query, params)
    rows = result.fetchall()

    return {
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "total_species": len(rows),
        "species": [
            {
                "scientific_name": row.scientific_name,
                "common_name_es": row.common_name_es,
                "common_name_en": row.common_name_en,
                "family": row.family,
                "iucn_status": row.iucn_status,
                "is_endemic": row.is_endemic,
                "is_migratory": row.is_migratory,
                "n_observations": int(row.n_observations),
                "elevation": {
                    "avg": (
                        round(float(row.avg_elevation), 1)
                        if row.avg_elevation
                        else None
                    ),
                    "min": int(row.min_elevation) if row.min_elevation else None,
                    "max": int(row.max_elevation) if row.max_elevation else None,
                },
                "centroid": {
                    "lat": (
                        round(float(row.centroid_lat), 6) if row.centroid_lat else None
                    ),
                    "lon": (
                        round(float(row.centroid_lon), 6) if row.centroid_lon else None
                    ),
                },
            }
            for row in rows
        ],
    }


@router.get("/filters")
async def get_available_filters(
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna los valores disponibles para los filtros del mapa.
    El frontend los usa para poblar los dropdowns.
    """
    families_query = text(
        """
        SELECT DISTINCT f.scientific_name
        FROM families f
        JOIN species s ON s.family_id = f.id
        JOIN recordings r ON r.species_id = s.id
        WHERE r.location IS NOT NULL
        ORDER BY f.scientific_name
    """
    )

    species_query = text(
        """
        SELECT DISTINCT s.scientific_name, s.common_name_es
        FROM species s
        JOIN recordings r ON r.species_id = s.id
        WHERE r.location IS NOT NULL
        ORDER BY s.scientific_name
        LIMIT 500
    """
    )

    families_result = await db.execute(families_query)
    species_result = await db.execute(species_query)

    return {
        "families": [row.scientific_name for row in families_result.fetchall()],
        "species": [
            {
                "scientific_name": row.scientific_name,
                "common_name_es": row.common_name_es,
            }
            for row in species_result.fetchall()
        ],
        "months": [
            {"value": i, "label": label}
            for i, label in enumerate(
                [
                    "Enero",
                    "Febrero",
                    "Marzo",
                    "Abril",
                    "Mayo",
                    "Junio",
                    "Julio",
                    "Agosto",
                    "Septiembre",
                    "Octubre",
                    "Noviembre",
                    "Diciembre",
                ],
                1,
            )
        ],
    }
