#!/usr/bin/env python3
"""
Actualización del pipeline para la aplicación web.  -- versión 1.0

Ejecuta, en orden, las 5 capas del proyecto y deja en data/ los archivos que lee la aplicación (app.py):
    1. construir_dataset_dolar_blue.py  → data/dataset_dolar_blue.csv
    2. diagnostico_dataset.py           → data/diagnostico_dataset.csv
    3. analisis_eda_features.py         → data/features_dolar_blue.csv (+ prueba de que ninguna variable usa datos futuros)
    4. comparar_modelos.py              → data/modelos/ (importancia de variables; solo si se pide o si todavía no existe)
    5. modelo_diario.py, h = 1 y h = 10 → data/modelo_h1/ y data/modelo_h10/ (predicción, rangos y alerta de Isolation Forest)
Al final escribe data/app/estado.json (fecha de actualización, último dato, versiones y controles).

Si algo falla (por ejemplo, el MEP o el CCL vienen vacíos o aparece una variable con datos futuros), el programa termina con
error y NO se publican resultados nuevos: la aplicación sigue mostrando los de la última actualización correcta.

Uso:
    python actualizar_pipeline.py                 # descarga todo y actualiza
    python actualizar_pipeline.py --comparar      # además vuelve a comparar algoritmos (≈ 2-3 min; lo hace los lunes)
    python actualizar_pipeline.py --sin-descarga  # usa el dataset que ya está en data/ (para probar sin internet)
En Colab (desde la carpeta del proyecto):
    !python actualizar_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "1.0"
HORIZONTES_APP = (1, 10)                 # 20 días no se publica: en 2024-2026 fue peor que el ingenuo
DIR_DATA = Path("data")
ESTADO = DIR_DATA / "app" / "estado.json"
ZONA = ZoneInfo("America/Argentina/Buenos_Aires")
MAX_DIAS_SIN_DATO = 5                    # días corridos: si el último blue es más viejo, el dato se considera desactualizado


def _paso(n: int, texto: str) -> None:
    print("\n" + "█" * 100 + f"\nPASO {n}: {texto}\n" + "█" * 100, flush=True)


def _validar_dataset(df: pd.DataFrame) -> list[str]:
    """Controles mínimos antes de modelar. Devuelve avisos (no detienen); los errores graves lanzan excepción."""
    avisos = []
    for col in ("blue_venta", "dolar_oficial", "dolar_mep", "dolar_ccl"):
        if col not in df.columns or df[col].tail(10).isna().all():
            raise RuntimeError(f"{col} no tiene datos recientes: no se puede predecir. Revisá la fuente en construir_dataset_dolar_blue.py.")
    ultimo = df.dropna(subset=["blue_venta"])["fecha"].max()
    dias = (pd.Timestamp(datetime.now(ZONA).date()) - ultimo).days
    if dias > MAX_DIAS_SIN_DATO:
        avisos.append(f"El último dato del blue es del {ultimo:%d/%m/%Y} ({dias} días): la fuente podría no estar actualizándose.")
    habiles = df[df["fecha"].dt.dayofweek < 5]
    for col in ("dolar_mep", "dolar_ccl"):
        s = habiles[col].dropna().tail(6)
        if len(s) == 6 and s.nunique() == 1:
            avisos.append(f"{col} repite el mismo valor en los últimos 6 días hábiles: posible hueco de la fuente.")
    ult = habiles.iloc[-1]
    for col, nombre in (("tasa_real_mensual", "la tasa real (BCRA o inflación)"), ("icc", "el ICC"), ("icg", "el ICG"),
                        ("inflacion_mensual_pct", "la inflación")):
        if pd.isna(ult.get(col)):
            avisos.append(f"Falta {nombre} en el último día: el modelo usa un valor típico en su lugar. Revisá la fuente o raw/.")
    return avisos


def main(comparar: bool = False, sin_descarga: bool = False) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    estado = {"version_actualizador": VERSION, "inicio": datetime.now(ZONA).isoformat(timespec="seconds"), "avisos": []}

    import construir_dataset_dolar_blue as m
    import diagnostico_dataset as d
    import analisis_eda_features as a
    import comparar_modelos as c
    import modelo_diario as md

    _paso(1, "construcción del dataset" + (" (OMITIDO: se usa el dataset existente)" if sin_descarga else ""))
    if sin_descarga:
        df = a._leer_csv(DIR_DATA / "dataset_dolar_blue.csv")
    else:
        df = m.main()
        df["fecha"] = pd.to_datetime(df["fecha"])
    estado["avisos"] += _validar_dataset(df)

    _paso(2, "diagnóstico de calidad")
    d.diagnosticar(str(DIR_DATA / "dataset_dolar_blue.csv"))

    _paso(3, "análisis exploratorio e ingeniería de variables")
    f, info = a.main(str(DIR_DATA / "dataset_dolar_blue.csv"))
    if not info["sin_fuga"]:
        raise RuntimeError("verificar_sin_fuga falló: alguna variable usa información futura. No se publican resultados.")

    imp = DIR_DATA / "modelos" / "importancia_variables.csv"
    if comparar or not imp.exists():
        _paso(4, "comparación de algoritmos y selección de variables")
        c.main(str(DIR_DATA / "dataset_dolar_blue.csv"))
    else:
        _paso(4, "comparación de algoritmos (OMITIDA: se reutiliza la última; usar --comparar para repetirla)")

    _paso(5, "predicción diaria")
    estado["predicciones"] = {}
    for h in HORIZONTES_APP:
        md.DIR_MODELO = DIR_DATA / f"modelo_h{h}"
        r = md.main(str(DIR_DATA / "dataset_dolar_blue.csv"), h=h)
        s = r["proximo_dia"].iloc[0]
        if not np.isfinite(s["blue_predicho"]):
            raise RuntimeError(f"La predicción a {h} día(s) no es un número válido.")
        estado["predicciones"][str(h)] = {"fecha_predicha": f"{s['fecha_predicha']:%d/%m/%Y}", "blue_predicho": round(float(s["blue_predicho"]), 2),
                                          "alerta_alta_incertidumbre": int(s["alerta_alta_incertidumbre"])}

    ult = df[df["fecha"].dt.dayofweek < 5].dropna(subset=["blue_venta"]).iloc[-1]     # último día hábil
    estado.update({
        "actualizado": datetime.now(ZONA).isoformat(timespec="seconds"),
        "ultimo_dato": f"{ult['fecha']:%d/%m/%Y}",
        "blue_ultimo_dato": float(ult["blue_venta"]),
        "versiones": {"construir_dataset": m.VERSION, "analisis_eda_features": a.VERSION,
                      "comparar_modelos": c.VERSION, "modelo_diario": md.VERSION},
        "grupos_modelo": list(c.GRUPOS_MODELO),
        "duracion_min": round((time.time() - t0) / 60, 1),
    })
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Pipeline completo en {estado['duracion_min']} min. Estado guardado en {ESTADO}")
    for x in estado["avisos"]:
        print("  ⚠", x)
    return estado


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Actualiza los datos y las predicciones que muestra la aplicación.")
    ap.add_argument("--comparar", action="store_true", help="volver a comparar algoritmos (importancia de variables)")
    ap.add_argument("--sin-descarga", action="store_true", help="no descargar: usar data/dataset_dolar_blue.csv")
    args = ap.parse_args()
    try:
        main(comparar=args.comparar, sin_descarga=args.sin_descarga)
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ La actualización falló: {e}\n  La aplicación sigue mostrando los resultados de la última actualización correcta.")
        sys.exit(1)
