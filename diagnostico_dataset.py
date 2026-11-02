#!/usr/bin/env python3
"""
Diagnóstico del dataset del dólar blue: ¿qué adecuaciones necesita antes de modelar?

Uso en Colab (misma carpeta que construir_dataset_dolar_blue.py):
    import diagnostico_dataset as d
    df, informe = d.diagnosticar()                      # busca data/dataset_dolar_blue.csv
    df, informe = d.diagnosticar("/ruta/dataset_dolar_blue.csv")

Imprime 9 controles y devuelve (df, informe). Guarda data/diagnostico_dataset.csv (separador ";", coma decimal).
No modifica el dataset. Nada de esto usa datos futuros: son solo mediciones sobre lo que ya tenés.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OBJETIVO = "blue_venta"
DIARIAS = ["blue_venta", "dolar_oficial", "dolar_mep", "dolar_ccl", "brecha_pct", "tasa_plazo_fijo_minorista_tna",
           "tasa_badlar_pct", "reservas_millones_usd", "merval_cierre", "merval_var_pct", "argt_var_pct", "riesgo_pais_pb"]
MENSUALES = ["inflacion_mensual_pct", "icg", "icc"]
TASAS = ["tasa_plazo_fijo_minorista_tna", "tasa_badlar_pct"]
# Columnas que YA son una variación (%): no se les vuelve a calcular log-retorno ni diferencia
VARIACIONES = ["merval_var_pct", "argt_var_pct"]
# Variables que se mueven con la inflación/devaluación (en pesos): conviene usar variaciones, no niveles
EN_PESOS = ["blue_venta", "dolar_oficial", "dolar_mep", "dolar_ccl"]


def _cargar(ruta):
    """Lee el CSV en el formato del proyecto (; + coma decimal + fechas DD/MM/AAAA); acepta también el formato viejo."""
    ruta = Path(ruta)
    with open(ruta, encoding="utf-8-sig") as f:
        enc = f.readline()
    sep, dec = (";", ",") if ";" in enc else (",", ".")
    df = pd.read_csv(ruta, sep=sep, decimal=dec, encoding="utf-8-sig")
    primera = str(df["fecha"].dropna().iloc[0])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y" if "/" in primera else "%Y-%m-%d")
    return df.sort_values("fecha").reset_index(drop=True)


def _buscar():
    for c in (Path("data/dataset_dolar_blue.csv"), Path("dataset_dolar_blue.csv")):
        if c.is_file():
            return c
    raise FileNotFoundError("No encontré dataset_dolar_blue.csv. Pasá la ruta: diagnosticar('/ruta/dataset_dolar_blue.csv')")


def _racha_final(s: pd.Series) -> int:
    """Cuántas filas consecutivas, hasta el final, repiten exactamente el último valor (ignora NaN al final)."""
    v = s.dropna()
    if v.empty:
        return 0
    iguales = (v.values[::-1] == v.values[-1])
    return int(np.argmin(iguales)) if not iguales.all() else int(len(v))


def _racha_max(s: pd.Series) -> int:
    v = s.dropna()
    if v.empty:
        return 0
    cambio = (v != v.shift()).cumsum()
    return int(v.groupby(cambio).size().max())


def _adf(x: pd.Series):
    try:
        from statsmodels.tsa.stattools import adfuller
        x = x.dropna()
        if len(x) < 50 or x.nunique() < 5:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(x, autolag="AIC")[1])
    except Exception:  # noqa: BLE001  (statsmodels ausente o serie degenerada)
        return np.nan


def diagnosticar(ruta=None, guardar=True):
    ruta = Path(ruta) if ruta else _buscar()
    df = _cargar(ruta)
    cols = [c for c in df.columns if c != "fecha"]
    n = len(df)
    print(f"Dataset: {ruta} | {n} filas | {df['fecha'].min():%d/%m/%Y} → {df['fecha'].max():%d/%m/%Y}")
    dias_entre = df["fecha"].diff().dt.days.dropna()
    print(f"Separación entre filas: mediana {dias_entre.median():.0f} día(s), máxima {dias_entre.max():.0f} → "
          f"{'calendario diario completo' if dias_entre.max() <= 1 else 'hay saltos (fines de semana/feriados o huecos)'}\n")

    # ───── 1. Resumen: escala, faltantes, asimetría, valores "congelados" ─────
    filas = []
    for c in cols:
        s = df[c]
        v = s.dropna()
        pos = v[v > 0]
        filas.append({
            "variable": c,
            "faltante_%": round(s.isna().mean() * 100, 1),
            "min": v.min() if len(v) else np.nan, "mediana": v.median() if len(v) else np.nan, "max": v.max() if len(v) else np.nan,
            "max/min": (pos.max() / pos.min()) if len(pos) and pos.min() > 0 else np.nan,
            "asimetria": v.skew() if len(v) > 3 else np.nan,
            "racha_final_igual_filas": _racha_final(s),
            "racha_max_igual_filas": _racha_max(s),
            "ultimo_valor": v.iloc[-1] if len(v) else np.nan,
        })
    res = pd.DataFrame(filas).set_index("variable")
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda x: f"{x:,.2f}")
    print("1) RESUMEN POR VARIABLE  (max/min = cuántas veces cambió la escala; racha = filas seguidas con el mismo valor)")
    print(res.to_string(), "\n")

    # ───── 2. Valores congelados al final (serie discontinuada o fuente caída) ─────
    print("2) VALORES CONGELADOS AL FINAL (una serie diaria que no cambia hace 8+ filas suele estar discontinuada o con la fuente caída)")
    alertas = []
    for c in DIARIAS:
        if c in df.columns:
            r = res.loc[c, "racha_final_igual_filas"]
            if r >= 8:                       # fin de semana + feriados largos llegan a ~5 filas iguales; 8 o más es sospechoso
                alertas.append((c, r, df[c].dropna().iloc[-1]))
    for c in MENSUALES:
        if c in df.columns and res.loc[c, "racha_final_igual_filas"] >= 70:
            alertas.append((c, res.loc[c, "racha_final_igual_filas"], df[c].dropna().iloc[-1]))
    if alertas:
        for c, r, v in alertas:
            print(f"   ⚠ {c}: las últimas {int(r)} filas valen exactamente {v:,.2f} (desde ≈ {df['fecha'].iloc[-int(r)]:%d/%m/%Y}).")
    else:
        print("   ✓ ninguna variable diaria lleva 8+ filas seguidas con el mismo valor.")
    for c in cols:                              # series que terminan en vacío (p. ej. falta raw/merval_msci.csv)
        validos = df[c].dropna()
        if c in DIARIAS and len(validos) and (df["fecha"].iloc[-1] - df.loc[validos.index[-1], "fecha"]).days > 4:
            print(f"   ⚠ {c}: sin datos desde {df.loc[validos.index[-1], 'fecha']:%d/%m/%Y} (último valor con dato). "
                  "Si es el Merval, cargá los cierres del MSCI Merval en raw/merval_msci.csv.")
    print()

    # ───── 3. Estacionariedad (ADF): niveles vs. variaciones ─────
    print("3) ESTACIONARIEDAD (ADF; p-valor < 0,05 ⇒ estacionaria). Una serie en NIVEL con p alto no sirve como objetivo directo.")
    filas = []
    for c in cols:
        s = df[c]
        if s.notna().sum() < 100:
            continue
        positiva = (s.dropna() > 0).all()
        if c in VARIACIONES:                     # ya es una variación: se evalúa tal cual
            filas.append({"variable": c, "p_nivel": _adf(s), "p_variacion": np.nan, "variacion_usada": "ya es variación"})
            continue
        delta = np.log(s).diff() if positiva and c not in TASAS and c not in ("brecha_pct",) else s.diff()
        filas.append({"variable": c, "p_nivel": _adf(s), "p_variacion": _adf(delta),
                      "variacion_usada": "log-retorno" if positiva and c not in TASAS and c != "brecha_pct" else "diferencia"})
    adf = pd.DataFrame(filas).set_index("variable")
    print(adf.to_string(float_format=lambda x: f"{x:.4f}") if not adf.empty else "   (statsmodels no disponible: pip install statsmodels)")
    print()

    # ───── 4. Saltos bruscos (devaluaciones, cambios de régimen, errores de fuente) ─────
    print("4) SALTOS DIARIOS MÁS GRANDES DEL BLUE Y DEL OFICIAL (%). Son reales si coinciden con eventos (p. ej. devaluación, fin del cepo).")
    for c in ("blue_venta", "dolar_oficial"):
        if c in df.columns:
            var = (df.set_index("fecha")[c].pct_change() * 100).dropna()
            top = var.reindex(var.abs().sort_values(ascending=False).index).head(6)
            print(f"   {c}: " + " | ".join(f"{f:%d/%m/%Y} {v:+.1f}%" for f, v in top.items()))
    if "brecha_pct" in df.columns:
        b = df["brecha_pct"].dropna()
        print(f"   brecha_pct: mín {b.min():.1f}% | máx {b.max():.1f}% | filas con brecha < 0: {(b < 0).sum()} (blue por debajo del oficial)")
    print()

    # ───── 5. Cambio de escala por inflación: ¿el rango de entrenamiento cubre el de prueba? ─────
    print("5) EXTRAPOLACIÓN: los árboles (XGBoost) no predicen fuera del rango visto. Valor del blue por año:")
    por_anio = df.groupby(df["fecha"].dt.year)[OBJETIVO].agg(["min", "max", "last"])
    print(por_anio.to_string())
    corte = int(n * 0.8)
    rng_train = (df[OBJETIVO].iloc[:corte].min(), df[OBJETIVO].iloc[:corte].max())
    fuera = (df[OBJETIVO].iloc[corte:] > rng_train[1]).mean() * 100
    print(f"   Con un corte 80/20 por fecha, el {fuera:.0f}% de las filas de prueba tiene un blue MAYOR al máximo de entrenamiento "
          f"({rng_train[1]:,.0f}).\n")

    # ───── 6. Colinealidad: tasas y variables en pesos ─────
    print("6) COLINEALIDAD (correlación de las VARIACIONES diarias; en niveles casi todo se correlaciona por tendencia común)")
    tas = [c for c in TASAS if c in df.columns]
    if len(tas) >= 2:
        cn = df[tas].corr()
        pares = [(a, b, cn.loc[a, b]) for i, a in enumerate(tas) for b in tas[i + 1:] if pd.notna(cn.loc[a, b])]
        print("   Tasas entre sí (nivel): " + " | ".join(f"{a.replace('tasa_', '')}~{b.replace('tasa_', '')} {r:.2f}" for a, b, r in pares))
    nivel = df[cols].corr(method="spearman")[OBJETIVO].drop(OBJETIVO).sort_values(key=abs, ascending=False)
    logs = df.set_index("fecha")[cols].copy()
    for c in logs.columns:
        if c in VARIACIONES:
            continue                               # ya es una variación
        logs[c] = np.log(logs[c]).diff() if (logs[c].dropna() > 0).all() and c not in TASAS else logs[c].diff()
    vari = logs.corr(method="spearman")[OBJETIVO].drop(OBJETIVO)
    comp = pd.DataFrame({"corr_NIVEL_vs_blue": nivel, "corr_VARIACION_vs_blue": vari}).reindex(nivel.index)
    print("   Correlación con el blue (Spearman): nivel vs. variación diaria")
    print(comp.to_string(float_format=lambda x: f"{x:+.2f}"))
    print("   → Si la correlación en nivel es alta y en variación ≈ 0, es tendencia compartida, no relación explicativa.\n")

    # ───── 7. Coherencia de unidades ─────
    print("7) COHERENCIA DE UNIDADES")
    for c in TASAS:
        if c in df.columns and df[c].notna().any():
            m = df[c].max()
            print(f"   {c}: máx {m:,.1f} → {'OK (% TNA)' if m > 1.5 else '⚠ parece estar en fracción (0,xx), no en %'}")
    if "reservas_millones_usd" in df.columns and df["reservas_millones_usd"].notna().any():
        print(f"   reservas_millones_usd: rango {df['reservas_millones_usd'].min():,.0f} – {df['reservas_millones_usd'].max():,.0f} (millones de USD)")
    if "icg" in df.columns and df["icg"].notna().any():
        print(f"   icg: {df['icg'].min():.2f} – {df['icg'].max():.2f} (escala 0–5) | icc: {df['icc'].min():.1f} – {df['icc'].max():.1f} (escala 0–100)")
    if "merval_cierre" in df.columns:
        u = df[["fecha", "merval_cierre"]].dropna().tail(10)
        print("   merval_cierre, últimas 10 filas (revisá si hay un salto el 19/10/2026, cambio de administrador S&P → MSCI):")
        print("      " + " | ".join(f"{f:%d/%m} {v:,.0f}" for f, v in zip(u["fecha"], u["merval_cierre"])))
    if "merval_var_pct" in df.columns and df["merval_var_pct"].notna().any():
        u = df[["fecha", "merval_var_pct"]].dropna().tail(10)
        print("   merval_var_pct (variación diaria %, empalme S&P → MSCI), últimas 10 filas con dato "
              "(0,00 = fin de semana o feriado):")
        print("      " + " | ".join(f"{f:%d/%m} {v:+.2f}" for f, v in zip(u["fecha"], u["merval_var_pct"])))
        print(f"      máx. |variación| diaria: {df['merval_var_pct'].abs().max():.1f}%  "
              "(un valor > 15 % suele indicar un error de empalme o de carga)")

    # ───── 8. Resolución real de los datos: ¿cuántos días hábiles NO cambian el valor? ─────
    print("\n8) DÍAS HÁBILES SIN CAMBIO RESPECTO DEL DÍA HÁBIL ANTERIOR, por año (%). Cerca de 100% = serie casi escalonada (poca resolución).")
    hab = df[df["fecha"].dt.dayofweek < 5].set_index("fecha")
    tabla = {}
    for c in ("blue_venta", "dolar_oficial", "dolar_mep", "dolar_ccl", "reservas_millones_usd"):
        if c in hab.columns:
            tabla[c] = ((hab[c].diff() == 0).groupby(hab.index.year).mean() * 100).round(0)
    print(pd.DataFrame(tabla).T.to_string())
    b = df.set_index("fecha")[OBJETIVO]
    enteros = (b % 1 == 0).mean() * 100
    print(f"   Valores del blue que son números enteros: {enteros:.0f}%"
          f"{'  → solo pesos enteros: con el blue en ~14-20 pesos (2016-2018) eso es una resolución muy baja' if enteros > 95 else ''}")
    fin_semana = (df["fecha"].dt.dayofweek >= 5).mean() * 100
    print(f"   El {fin_semana:.0f}% de las filas son sábados/domingos (copias del viernes): conviene modelar solo días hábiles.\n")

    # ───── 9. Modelo ingenuo "dentro de h días el blue vale lo mismo que hoy": ¿cuánto MAPE da? ─────
    print("9) MAPE DEL MODELO INGENUO (blue[t+h] = blue[t]); es el piso que cualquier modelo tiene que superar")
    for h in (1, 7, 15, 30):
        y = b.shift(-h)
        todo = (abs(y - b) / y).dropna() * 100
        rec = todo[todo.index >= "2024-01-01"]
        print(f"   h = {h:>2} días corridos: {todo.mean():5.2f}% (todo el período) | {rec.mean():5.2f}% (desde 2024)")
    print("   → Si este número ya es menor que tu umbral de MAPE (10%), el umbral no demuestra que el modelo sirva.")

    informe = res.join(adf, how="left")
    if guardar:
        Path("data").mkdir(exist_ok=True)
        informe.to_csv("data/diagnostico_dataset.csv", sep=";", decimal=",", encoding="utf-8-sig")
        print("\nGuardado: data/diagnostico_dataset.csv")
    return df, informe


if __name__ == "__main__":
    diagnosticar()
