#!/usr/bin/env python3
"""
Análisis exploratorio (EDA) + ingeniería de variables para predecir el dólar blue.  -- versión 1.0

Entrada : data/dataset_dolar_blue.csv  (lo genera construir_dataset_dolar_blue.py; también acepta dataset_dolar_blue_4.csv, etc.)
Salidas : data/features_dolar_blue.csv      tabla lista para modelar (variables + objetivos + columna "conjunto")
          data/eda/ranking_variables.csv    relevancia de cada variable SOLO con datos de entrenamiento
          data/eda/baselines_test.csv       errores de los modelos "tontos" que cualquier modelo debe superar
          data/eda/rezagos_diarios.csv      correlación variable(t-L) vs. variación futura del blue
          data/eda/rezagos_mensuales.csv    ICG / ICC / inflación vs. variación del blue de los meses siguientes (hipótesis H1)
          data/eda/01..05_*.png             gráficos

Decisiones de diseño (todas se pueden cambiar en CONFIGURACIÓN):
  1. Se modela sobre DÍAS HÁBILES (lun-vie). Los sábados/domingos son copias del viernes y duplicarían filas idénticas.
  2. OBJETIVO = variación logarítmica del blue a h días hábiles:  y_h = 100 * ln(blue[t+h] / blue[t]).
     No se predice el nivel: el blue pasó de 14 a 1.570 pesos y un árbol (XGBoost) no puede predecir valores fuera del rango visto.
     El precio se reconstruye así:  blue_predicho = blue[t] * exp(y_predicho / 100)  y el MAPE se mide sobre ese precio.
  3. Ninguna variable usa datos futuros: en la fila t solo entra información disponible hasta t (se verifica solo con
     verificar_sin_fuga()). Las variables del BCRA ya llegan con 1 día de rezago desde el dataset.
  4. ICG, ICC e inflación se usan con rezago de 1 mes (t-1): el valor que se conocía hace un mes.
  5. Corte cronológico: entrenamiento → purga (h días sin usar, para que el objetivo no se solape) → test (últimos DIAS_HOLDOUT
     días hábiles, NO se toca hasta el final). Para ajustar hiperparámetros: walk_forward() (ventana creciente, con purga).
  6. Todo el análisis de relevancia (correlaciones, información mutua, colinealidad, Isolation Forest) usa SOLO entrenamiento.

FORMATO DE LOS CSV (igual que el resto del proyecto): separador ";", COMA decimal y fechas DD/MM/AAAA.
    pandas : df = pd.read_csv("data/features_dolar_blue.csv", sep=";", decimal=",", encoding="utf-8-sig")
             df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
             (o simplemente  a.leer_features()  )
    R      : df <- read.csv2("data/features_dolar_blue.csv", fileEncoding = "UTF-8-sig"); df$fecha <- as.Date(df$fecha, format = "%d/%m/%Y")
    Excel / Power BI: delimitador "punto y coma", origen UTF-8; si los decimales salen mal, configuración regional Español (Argentina).

Uso en Colab:
    !pip install -q scikit-learn scipy matplotlib
    import analisis_eda_features as a
    f, info = a.main()                          # corre todo (usa data/dataset_dolar_blue*.csv)
    f, info = a.main("/ruta/dataset_dolar_blue.csv")
    X_tr, y_tr, X_te, y_te = a.preparar_xy(f, info["columnas_modelo"], h=10)     # para el modelado
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2.0"

# ───────────────────────────── CONFIGURACIÓN ─────────────────────────────
DIR_OUT = Path("data")
DIR_FIG = DIR_OUT / "eda"
CSV_SEP, CSV_DECIMAL, CSV_FECHA = ";", ",", "%d/%m/%Y"

HORIZONTES = (1, 5, 10, 20)     # días hábiles hacia adelante: 1 = "hoy" (con datos hasta ayer); 10 ≈ 14 días corridos; 20 ≈ 28. Hipótesis H1: 15-30 días corridos
H_PRINCIPAL = 10                # horizonte usado para el ranking de variables, el análisis de rezagos y el split
FECHA_INICIO_MODELO = "2016-04-01"   # las primeras filas no tienen ventanas completas (60-252 días); poné "2020-01-01" para descartar la era previa
DIAS_HOLDOUT = 120              # días hábiles finales reservados para el test (≈ 6 meses)
UMBRAL_BRECHA_ALTA = 20.0       # brecha (%) a partir de la cual se considera "régimen de cepo"; el régimen actual empieza después de la última vez que se superó
CORR_MAX = 0.95                 # |Spearman| máxima entre dos variables; por encima se descarta la menos informativa
CONTAMINACION_IFOREST = 0.02    # proporción de días que Isolation Forest marca como atípicos (se MARCAN, no se borran)
SEMILLA = 42

COL_BASE = ["fecha", "blue_venta", "dolar_oficial", "dolar_mep", "dolar_ccl", "brecha_pct", "tasa_real_mensual", "tasa_badlar_pct",
            "inflacion_mensual_pct", "reservas_millones_usd", "riesgo_pais_pb", "icg", "icc"]
COL_OPCIONALES = ["merval_var_pct", "argt_var_pct"]
# Dólar MEP y CCL: forman parte del modelo SIEMPRE (grupo E). El período de modelado empieza el primer día en que existen
# todas estas variables; antes de esa fecha no se rellenan años sin datos.
VARIABLES_MEP_CCL = ["brecha_blue_mep_pct", "brecha_blue_mep_cambio_5", "ret_mep_1", "ret_mep_5", "ret_mep_20",
                     "brecha_blue_ccl_pct", "brecha_blue_ccl_cambio_5", "ret_ccl_1", "ret_ccl_5", "ret_ccl_20", "canje_ccl_mep_pct"]

# ── Régimen cambiario y calendario político (fechas conocidas de antemano: no son información futura) ──
# Cepo para personas: DNU 609/2019 (01/09/2019) hasta el levantamiento para personas humanas del 14/04/2025.
# (Entre 2011 y el 16/12/2015 también hubo cepo, pero el dataset empieza en 2016.)
CEPO_PERIODOS = [("2019-09-01", "2025-04-13")]
# Elecciones nacionales (y la provincial bonaerense de 2025, que movió el mercado). En 2025 no hubo PASO nacionales.
ELECCIONES = ["2017-08-13", "2017-10-22", "2019-08-11", "2019-10-27", "2021-09-12", "2021-11-14",
              "2023-08-13", "2023-10-22", "2023-11-19", "2025-09-07", "2025-10-26"]
TOPE_DIAS_ELECCION = 60          # más allá de 60 días hábiles, la elección "no está cerca"
# Variables "de nivel lento" (cambian poco en el tiempo): su información mutua suele reflejar EN QUÉ ÉPOCA estamos, no capacidad predictiva
NIVELES_LENTOS = ["icg_t1", "icc_t1", "infl_t1", "rp_nivel", "badlar_nivel", "tasa_real_mensual", "brecha_pct", "brecha_z_120",
                  "reservas_vs_max_252", "regimen_brecha_alta", "vol_blue_60",
                  "d_icg_t1", "d_icc_t1", "infl_acel_t1"]      # mensuales: constantes dentro del mes, también identifican la época


# ───────────────────────────── LECTURA ─────────────────────────────
def _buscar() -> Path:
    fijos = [Path("data/dataset_dolar_blue.csv"), Path("dataset_dolar_blue.csv")]
    globs = [p for carpeta in (Path("data"), Path(".")) for p in sorted(carpeta.glob("dataset_dolar_blue*.csv"))
             if "backup" not in p.name.lower()]
    for c in fijos + globs:
        if c.is_file():
            return c
    raise FileNotFoundError("No encontré dataset_dolar_blue.csv. Pasá la ruta: main('/ruta/dataset_dolar_blue.csv')")


def _leer_csv(ruta) -> pd.DataFrame:
    """Lee un CSV del proyecto: ';' + coma decimal + fechas DD/MM/AAAA (también el formato viejo ',' + '.' + AAAA-MM-DD)."""
    ruta = Path(ruta)
    with open(ruta, encoding="utf-8-sig") as fh:
        enc = fh.readline()
    sep, dec = (";", ",") if ";" in enc else (",", ".")
    df = pd.read_csv(ruta, sep=sep, decimal=dec, encoding="utf-8-sig")
    primera = str(df["fecha"].dropna().iloc[0])
    df["fecha"] = pd.to_datetime(df["fecha"], format=CSV_FECHA if "/" in primera else "%Y-%m-%d")
    return df.sort_values("fecha").drop_duplicates("fecha").reset_index(drop=True)


def leer_features(ruta=None) -> pd.DataFrame:
    """Lee data/features_dolar_blue.csv con el formato correcto."""
    return _leer_csv(ruta or DIR_OUT / "features_dolar_blue.csv")


def _guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime(CSV_FECHA)
    df.to_csv(ruta, index=False, sep=CSV_SEP, decimal=CSV_DECIMAL, encoding="utf-8-sig")


def _titulo(txt: str) -> None:
    print("\n" + "═" * 100 + f"\n{txt}\n" + "═" * 100)


# ───────────────────────────── 1. CALIDAD DE DATOS ─────────────────────────────
def informe_calidad(d_all: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    _titulo("1) CALIDAD DE DATOS")
    print(f"Filas: {len(d_all)} ({d_all['fecha'].min():%d/%m/%Y} → {d_all['fecha'].max():%d/%m/%Y}) | "
          f"sábados/domingos: {(d_all['fecha'].dt.dayofweek >= 5).mean()*100:.0f}% → se descartan; quedan {len(d)} días hábiles.")
    falt = (d_all.isna().mean() * 100).round(1)
    falt = falt[falt > 0]
    print("Faltantes (%):", falt.to_dict() if len(falt) else "ninguno")
    for c in COL_OPCIONALES:
        if c in d_all.columns and d_all[c].notna().any():
            ult = d_all.loc[d_all[c].last_valid_index(), "fecha"]
            if (d_all["fecha"].iloc[-1] - ult).days > 4:
                print(f"  ⚠ {c}: sin datos desde {ult + pd.Timedelta(days=1):%d/%m/%Y}. Para el MSCI Merval cargá raw/merval_msci.csv "
                      "y volvé a correr construir_dataset_dolar_blue.py. Las variables derivadas quedan vacías (XGBoost admite NaN).")
    print(f"  Días con brecha < 0 (blue por debajo del oficial): {(d['brecha_pct'] < 0).sum()}")

    g = d.groupby(d["fecha"].dt.year)
    lb = np.log(d["blue_venta"])
    d2 = d.assign(r1=lb.diff() * 100, sin_cambio=(d["blue_venta"].diff() == 0) * 100, entero=(d["blue_venta"] % 1 == 0) * 100)
    g2 = d2.groupby(d2["fecha"].dt.year)
    tabla = pd.DataFrame({
        "blue_cierre": g["blue_venta"].last(), "brecha_media_%": g["brecha_pct"].mean(),
        "vol_diaria_blue_%": g2["r1"].std(), "inflacion_media_%": g["inflacion_mensual_pct"].mean(),
        "reservas_cierre": g["reservas_millones_usd"].last(), "riesgo_pais_medio": g["riesgo_pais_pb"].mean(),
        "%dias_blue_sin_cambio": g2["sin_cambio"].mean(), "%blue_entero": g2["entero"].mean()}).round(1)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nResumen por año (el blue cambia de escala y de régimen: de ~14 a ~1.570 pesos; brecha de ~2% a ~100% y de vuelta a ~1%):")
    print(tabla.to_string())
    top = (lb.diff() * 100).dropna()
    top = top.reindex(top.abs().sort_values(ascending=False).index).head(6)
    print("\nMayores saltos diarios del blue (%):", " | ".join(f"{f:%d/%m/%Y} {v:+.1f}" for f, v in zip(d.loc[top.index, "fecha"], top)))
    print("  → son eventos reales (PASO 2019, devaluación de dic-2023, etc.): NO se borran; se marcan con Isolation Forest y se usa un objetivo winsorizado opcional.")
    print("\nSobre normalización: el modelo NO recibe el blue, el oficial ni las reservas en nivel, sino variaciones (%), brecha y diferencias;")
    print("  por eso los cambios de escala (pesos, millones de USD, puntos básicos, %) no obligan a estandarizar para árboles (XGBoost).")
    print("  Para regresión lineal / redes sí: usar estandarizar(train, test) más abajo (ajustada solo con el entrenamiento).")
    return tabla


# ───────────────────────────── 2. INGENIERÍA DE VARIABLES ─────────────────────────────
def _hace_meses(d: pd.DataFrame, col: str, meses: int) -> pd.Series:
    """Valor que tenía `col` (según el dataset, ya con su fecha de publicación) hace `meses` meses calendario."""
    s = d.set_index("fecha")[col]
    objetivo = pd.DatetimeIndex(d["fecha"] - pd.DateOffset(months=meses))
    return pd.Series(s.reindex(objetivo, method="ffill").values, index=d.index)


def construir_features(d: pd.DataFrame) -> pd.DataFrame:
    """d: dataset en días hábiles. Devuelve fecha, blue_venta, variables y objetivos. Solo usa información hasta t en la fila t."""
    f = pd.DataFrame({"fecha": d["fecha"], "blue_venta": d["blue_venta"]})
    blue, ofi = d["blue_venta"], d["dolar_oficial"]
    lb, lo = np.log(blue), np.log(ofi)

    # ── Dinámica propia del blue (momentum, volatilidad, reversión a la media) ──
    for k in (1, 5, 10, 20, 60):
        f[f"ret_blue_{k}"] = (lb - lb.shift(k)) * 100
    r1 = lb.diff() * 100
    f["vol_blue_20"] = r1.rolling(20, min_periods=20).std()
    f["vol_blue_60"] = r1.rolling(60, min_periods=60).std()
    for k in (20, 60):
        f[f"dist_media_blue_{k}"] = (lb - lb.rolling(k, min_periods=k).mean()) * 100

    # ── Dólar oficial y brecha (la brecha es la señal de presión cambiaria) ──
    for k in (5, 20):
        f[f"ret_oficial_{k}"] = (lo - lo.shift(k)) * 100
    br = d["brecha_pct"]
    f["brecha_pct"] = br
    for k in (5, 20):
        f[f"brecha_cambio_{k}"] = br - br.shift(k)
    mb = br.rolling(120, min_periods=120)
    f["brecha_z_120"] = (br - mb.mean()) / mb.std().replace(0, np.nan)
    f["regimen_brecha_alta"] = (br > UMBRAL_BRECHA_ALTA).astype(int)

    # ── Dólar MEP y CCL (si el dataset los trae): brecha del blue contra los dólares financieros legales y su dinámica.
    #    Antes del primer dato de cada uno quedan VACÍAS (no se rellenan con nada inventado). ──
    for col, pref in (("dolar_mep", "mep"), ("dolar_ccl", "ccl")):
        if col in d.columns and d[col].notna().any():
            lx = np.log(d[col])
            f[f"brecha_blue_{pref}_pct"] = (lb - lx) * 100
            f[f"brecha_blue_{pref}_cambio_5"] = f[f"brecha_blue_{pref}_pct"] - f[f"brecha_blue_{pref}_pct"].shift(5)
            for k in (1, 5, 20):
                f[f"ret_{pref}_{k}"] = (lx - lx.shift(k)) * 100
    if {"dolar_mep", "dolar_ccl"} <= set(d.columns) and d["dolar_mep"].notna().any() and d["dolar_ccl"].notna().any():
        f["canje_ccl_mep_pct"] = (np.log(d["dolar_ccl"]) - np.log(d["dolar_mep"])) * 100

    # ── Reservas del BCRA: variación a 5, 10 y 20 días (recomendado) y distancia al máximo de 1 año ──
    res = d["reservas_millones_usd"]
    lr = np.log(res)
    for k in (5, 10, 20):
        f[f"var_reservas_{k}"] = (lr - lr.shift(k)) * 100
    f["reservas_vs_max_252"] = (lr - np.log(res.rolling(252, min_periods=60).max())) * 100

    # ── Riesgo país ──
    rp = d["riesgo_pais_pb"]
    f["rp_nivel"] = rp
    for k in (5, 20):
        f[f"rp_cambio_{k}"] = rp - rp.shift(k)

    # ── Tasas (se usa BADLAR; el plazo fijo se omite: correlación 0,998 con BADLAR) ──
    bad = d["tasa_badlar_pct"]
    f["badlar_nivel"] = bad
    f["badlar_cambio_5"] = bad - bad.shift(5)
    f["tasa_real_mensual"] = d["tasa_real_mensual"]
    f["carry_oficial_20"] = bad * 20 / 365 - f["ret_oficial_20"]      # rendimiento del peso (BADLAR a 20 días) menos la depreciación oficial de 20 días

    # ── Mercado de capitales: Merval y ARGT (variaciones diarias acumuladas; vacías si la fuente no tiene datos) ──
    for col, pref in (("merval_var_pct", "merval"), ("argt_var_pct", "argt")):
        if col in d.columns and d[col].notna().any():
            lv = np.log1p(d[col] / 100)
            for k in (5, 20):
                f[f"{pref}_ret_{k}"] = lv.rolling(k, min_periods=k).sum() * 100
    if "argt_var_pct" in d.columns and d["argt_var_pct"].notna().any():
        f["argt_vol_20"] = d["argt_var_pct"].rolling(20, min_periods=20).std()

    # ── Régimen (cepo) y calendario: se conocen de antemano ──
    fe = d["fecha"]
    f["cepo"] = 0
    for ini, fin in CEPO_PERIODOS:
        f.loc[(fe >= ini) & (fe <= fin), "cepo"] = 1
    # Días hábiles de CALENDARIO (lun-vie) hasta la próxima elección y desde la última: no dependen de qué filas hay en el dataset
    hoy = fe.values.astype("datetime64[D]")
    elec = np.array(sorted(ELECCIONES), dtype="datetime64[D]")
    sig = np.searchsorted(elec, hoy, side="right")              # próxima elección estrictamente posterior a hoy
    ant = sig - 1                                               # última elección anterior o igual a hoy
    hasta = np.where(sig < len(elec), np.busday_count(hoy, elec[np.minimum(sig, len(elec) - 1)]), TOPE_DIAS_ELECCION)
    desde = np.where(ant >= 0, np.busday_count(elec[np.maximum(ant, 0)], hoy), TOPE_DIAS_ELECCION)
    f["dias_a_eleccion"] = np.clip(hasta, 0, TOPE_DIAS_ELECCION)
    f["dias_desde_eleccion"] = np.clip(desde, 0, TOPE_DIAS_ELECCION)
    f["dia_semana"] = fe.dt.dayofweek
    f["dia_mes"] = fe.dt.day
    f["mes_aguinaldo"] = fe.dt.month.isin([6, 7, 12]).astype(int)        # aguinaldo: se cobra a fin de junio y de diciembre

    # ── Variables mensuales con rezago t-1 (valor de hace un mes) ──
    f["icg_t1"] = _hace_meses(d, "icg", 1)
    f["icc_t1"] = _hace_meses(d, "icc", 1)
    f["d_icg_t1"] = f["icg_t1"] - _hace_meses(d, "icg", 2)         # cambio mensual del índice, también rezagado
    f["d_icc_t1"] = f["icc_t1"] - _hace_meses(d, "icc", 2)
    f["infl_t1"] = _hace_meses(d, "inflacion_mensual_pct", 1)
    f["infl_acel_t1"] = f["infl_t1"] - _hace_meses(d, "inflacion_mensual_pct", 2)

    # ── Objetivos: variación futura (%) del blue a h días hábiles, y el precio futuro para medir el MAPE ──
    for h in HORIZONTES:
        f[f"y_{h}"] = (lb.shift(-h) - lb) * 100
        f[f"blue_t_mas_{h}"] = blue.shift(-h)
    return f


def verificar_sin_fuga(d: pd.DataFrame, n_cortes: int = 3) -> bool:
    """
    Prueba anti look-ahead: las variables de las filas <= T deben ser IDÉNTICAS si se borran todos los datos posteriores a T.
    Si alguna variable usara información futura, cambiaría. Devuelve True si pasa.
    """
    completo = construir_features(d)
    cols = [c for c in completo.columns if not c.startswith(("y_", "blue_t_mas_"))]
    ok = True
    for frac in np.linspace(0.45, 0.9, n_cortes):
        k = int(len(d) * frac)
        parcial = construir_features(d.iloc[:k].reset_index(drop=True))
        try:
            pd.testing.assert_frame_equal(completo.loc[:k - 1, cols].reset_index(drop=True), parcial[cols], check_exact=False, rtol=1e-9, atol=1e-9)
        except AssertionError as e:
            ok = False
            print(f"  ✗ FUGA DE DATOS FUTUROS detectada cortando en {d['fecha'].iloc[k-1]:%d/%m/%Y}: {str(e)[:200]}")
    print("  ✓ verificar_sin_fuga: las variables de cada día no cambian si se eliminan los datos posteriores." if ok else "")
    return ok


# ───────────────────────────── 3. CONJUNTOS (train / purga / test) ─────────────────────────────
def inicio_mep_ccl(f: pd.DataFrame) -> pd.Timestamp:
    """Primer día con TODAS las variables de MEP y CCL (incluidas las de 20 días). Antes de esa fecha no se modela."""
    faltan = [c for c in VARIABLES_MEP_CCL if c not in f.columns]
    if faltan:
        raise ValueError(f"Faltan las variables de MEP/CCL {faltan}: el dataset debe traer dolar_mep y dolar_ccl con datos "
                         "(corré construir_dataset_dolar_blue.py v3.4).")
    completas = f[VARIABLES_MEP_CCL].notna().all(axis=1)
    if not completas.any():
        raise ValueError("No hay ningún día con todas las variables de MEP y CCL: revisá dolar_mep y dolar_ccl en el dataset.")
    return max(pd.Timestamp(FECHA_INICIO_MODELO), f.loc[completas, "fecha"].min())


def asignar_conjuntos(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy().reset_index(drop=True)
    n, h_max = len(f), max(HORIZONTES)
    conj = np.full(n, "train", dtype=object)
    con_y = np.flatnonzero(f[f"y_{H_PRINCIPAL}"].notna().values)
    ini_test = int(con_y[-DIAS_HOLDOUT])
    conj[max(ini_test - h_max, 0):ini_test] = "purga"
    conj[ini_test:] = "test"
    conj[f[f"y_{H_PRINCIPAL}"].isna().values & (np.arange(n) >= ini_test)] = "sin_objetivo"   # últimos días: se pueden predecir pero aún no tienen resultado
    conj[(f["fecha"] < inicio_mep_ccl(f)).values] = "previo"     # calentamiento y años sin MEP/CCL
    f["conjunto"] = conj
    return f


def walk_forward(f: pd.DataFrame, n_splits: int = 6, test_size: int = 120, gap: int | None = None):
    """
    Validación para ajustar hiperparámetros: ventana creciente, siempre entrenando con el pasado y con una purga de `gap` días.
    Con 6 tramos de 120 días hábiles se valida sobre ≈ 3 años (incluye la devaluación de dic-2023 y la salida del cepo), no solo sobre
    el último período tranquilo: un modelo que solo anda en calma no sirve.
    """
    gap = gap if gap is not None else max(HORIZONTES)
    tr = np.flatnonzero((f["conjunto"] == "train").values)
    minimo = 500                                   # días mínimos de entrenamiento en el primer tramo
    if len(tr) - n_splits * test_size - gap < minimo:
        test_size = max(40, (len(tr) - gap - minimo) // n_splits)
        print(f"  (walk-forward: con {len(tr)} días de entrenamiento se usan tramos de {test_size} días)")
    for k in range(n_splits, 0, -1):
        v_fin = len(tr) - (k - 1) * test_size
        v_ini = v_fin - test_size
        yield tr[:v_ini - gap], tr[v_ini:v_fin]


def _inicio_regimen_actual(f: pd.DataFrame) -> pd.Timestamp:
    alta = f.loc[f["brecha_pct"] > UMBRAL_BRECHA_ALTA, "fecha"]
    return (alta.max() + pd.Timedelta(days=1)) if len(alta) else pd.Timestamp("2024-01-01")


# ───────────────────────────── 4. LÍNEAS BASE ─────────────────────────────
def evaluar_baselines(f: pd.DataFrame) -> pd.DataFrame:
    """Errores de predictores 'tontos' sobre el TEST. Un modelo solo vale si los supera."""
    _titulo("4) LÍNEAS BASE (el piso que tu modelo debe superar) — evaluadas en TEST")
    tr, te = f[f["conjunto"] == "train"], f[f["conjunto"] == "test"]
    print(f"Train: {tr['fecha'].min():%d/%m/%Y} → {tr['fecha'].max():%d/%m/%Y} ({len(tr)} días) | "
          f"Test: {te['fecha'].min():%d/%m/%Y} → {te['fecha'].max():%d/%m/%Y} ({len(te)} días)")
    filas = []
    for h in HORIZONTES:
        y = f"y_{h}"
        t = te.dropna(subset=[y])
        drift = tr[y].mean()
        preds = {"ingenuo: blue(t+h) = blue(t)": pd.Series(0.0, index=t.index),
                 f"deriva media del train ({drift:+.2f}%)": pd.Series(drift, index=t.index),
                 "momentum: repite el retorno pasado": t[f"ret_blue_{h}"] if f"ret_blue_{h}" in t else None}
        for nombre, yp in preds.items():
            if yp is None:
                continue
            pp = t["blue_venta"] * np.exp(yp / 100)
            real = t[f"blue_t_mas_{h}"]
            dirs = np.sign(yp[yp != 0]) == np.sign(t.loc[yp[yp != 0].index, y])
            filas.append({"horizonte_dias_habiles": h, "modelo": nombre,
                          "MAPE_%": (abs(pp - real) / real).mean() * 100, "MAE_pesos": (pp - real).abs().mean(),
                          "RMSE_pesos": float(np.sqrt(((pp - real) ** 2).mean())),
                          "acierto_direccion_%": dirs.mean() * 100 if len(dirs) else np.nan})
    res = pd.DataFrame(filas)
    print(res.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    ing = res[res["modelo"].str.startswith("ingenuo")]
    print("\n→ H2 (MAPE < 10%): el modelo ingenuo YA cumple " + ", ".join(f"h={int(r.horizonte_dias_habiles)}: {r['MAPE_%']:.2f}%" for _, r in ing.iterrows()) +
          ".\n  Por eso el umbral de 10% no demuestra que el modelo sirva: hay que mostrar que baja el MAPE/MAE respecto del ingenuo (y en qué %).")
    anio = (tr.assign(a=tr["fecha"].dt.year, e=lambda x: (np.exp(x[f"y_{H_PRINCIPAL}"] / 100) - 1).abs() * 100)
              .groupby("a")["e"].mean().round(2))
    print(f"\nMAPE del ingenuo a h={H_PRINCIPAL} por año (train): {anio.to_dict()}")
    print("  → con inflación y cepo altos el error del ingenuo es mucho mayor: evaluar el modelo POR PERÍODO, no solo en promedio.")
    return res


# ───────────────────────────── 5. RELEVANCIA DE VARIABLES (solo train) ─────────────────────────────
def columnas_features(f: pd.DataFrame) -> list[str]:
    excl = {"fecha", "blue_venta", "conjunto", "outlier_iforest"}
    return [c for c in f.columns if c not in excl and not c.startswith(("y_", "blue_t_mas_"))]


def ranking_variables(f: pd.DataFrame, ini_actual: pd.Timestamp) -> pd.DataFrame:
    _titulo(f"5) RELEVANCIA DE VARIABLES vs. y_{H_PRINCIPAL} (solo datos de entrenamiento)")
    cols = columnas_features(f)
    tr = f[(f["conjunto"] == "train")].dropna(subset=[f"y_{H_PRINCIPAL}"])
    y = tr[f"y_{H_PRINCIPAL}"]
    act = tr[tr["fecha"] >= ini_actual]
    print(f"Filas train: {len(tr)} (muestra efectiva ≈ {len(tr)//H_PRINCIPAL}, porque el objetivo a {H_PRINCIPAL} días se solapa) | "
          f"régimen actual (brecha < {UMBRAL_BRECHA_ALTA:.0f}% desde {ini_actual:%d/%m/%Y}): {len(act)} filas")
    X = tr[cols]
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X.fillna(X.median()).fillna(0), y, random_state=SEMILLA)
    except Exception as e:  # noqa: BLE001
        print("  (sin scikit-learn no se calcula la información mutua:", e, ")")
        mi = np.full(len(cols), np.nan)
    r = pd.DataFrame({"variable": cols,
                      "spearman_train": [X[c].corr(y, method="spearman") for c in cols],
                      "spearman_regimen_actual": [act[c].corr(act[f"y_{H_PRINCIPAL}"], method="spearman") for c in cols],
                      "info_mutua": mi, "faltante_%": (X.isna().mean() * 100).values})
    r = r.sort_values("info_mutua", ascending=False).reset_index(drop=True)

    # Colinealidad: se recorre de mayor a menor información mutua y se descarta la que se parece (> CORR_MAX) a una ya elegida
    cm = X.corr(method="spearman").abs()
    elegidas, motivo = [], {}
    for c in r["variable"]:
        par = next((e for e in elegidas if pd.notna(cm.loc[c, e]) and cm.loc[c, e] > CORR_MAX), None)
        if par is None or c in VARIABLES_MEP_CCL:          # MEP y CCL son obligatorias: nunca se descartan
            elegidas.append(c)
        else:
            motivo[c] = f"colineal con {par} (|ρ|={cm.loc[c, par]:.2f})"
    r["descartar"] = r["variable"].map(motivo).fillna("")
    ic_tr = 1.96 / np.sqrt(max(len(tr) // H_PRINCIPAL, 3))                    # cota aproximada con la muestra efectiva (objetivo solapado)
    ic_act = 1.96 / np.sqrt(max(len(act) // H_PRINCIPAL, 3))
    r["tipo"] = np.where(r["variable"].isin(NIVELES_LENTOS), "nivel lento", "dinámica")
    pd.set_option("display.width", 220)
    print("Variables de DINÁMICA (variaciones, retornos, volatilidad): las que pueden anticipar movimientos")
    print(r[r["tipo"] == "dinámica"].head(15).drop(columns="tipo").to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nVariables LENTAS o MENSUALES (ICC, ICG, inflación, riesgo país, tasas, brecha): ⚠ su información mutua es alta casi siempre porque")
    print("  identifican la ÉPOCA (p. ej. 2023 = alta inflación y alta volatilidad), no porque anticipen el blue. No las leas como poder predictivo.")
    print(r[r["tipo"] == "nivel lento"].drop(columns="tipo").to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nRuido: con la muestra efectiva, |ρ| < {ic_tr:.2f} (todo el train) y |ρ| < {ic_act:.2f} (régimen actual, solo ≈{len(act)//H_PRINCIPAL} observaciones")
    print("  independientes) no se distingue de cero. Las correlaciones simples con retornos son bajas por naturaleza: el valor está en")
    print("  combinarlas con árboles; usá este ranking para descartar ruido, no para elegir 'la variable'.")
    r["ruido_train_+-"], r["ruido_actual_+-"] = ic_tr, ic_act
    if motivo:
        print("Descartadas por colinealidad:", "; ".join(f"{k} ({v})" for k, v in motivo.items()))
    _guardar_csv(r, DIR_FIG / "ranking_variables.csv")
    return r


# ───────────────────────────── 6. REZAGOS ─────────────────────────────
def analisis_rezagos_diarios(f: pd.DataFrame, ini_actual: pd.Timestamp, lags=(0, 5, 10, 20, 40, 60)) -> dict[str, pd.DataFrame]:
    _titulo(f"6a) DESFASES DIARIOS: variable(t-L) vs. variación futura del blue y_{H_PRINCIPAL} (Spearman, solo train)")
    vars_ = [c for c in ("var_reservas_5", "rp_cambio_5", "badlar_cambio_5", "ret_oficial_5", "brecha_cambio_5", "merval_ret_5", "argt_ret_5", "ret_blue_5")
             if c in f.columns]
    es_train = (f["conjunto"] == "train").values
    y = f[f"y_{H_PRINCIPAL}"]
    out = {}
    for nombre, mask in (("todo el train", es_train), ("régimen actual", es_train & (f["fecha"] >= ini_actual).values)):
        m = pd.DataFrame({L: {v: f[v].shift(L)[mask].corr(y[mask], method="spearman") for v in vars_} for L in lags})
        m.columns = [f"L={L}" for L in lags]
        out[nombre] = m
        print(f"\n{nombre} (filas: {int(mask.sum())}):")
        print(m.to_string(float_format=lambda x: f"{x:+.3f}"))
    filas = [m.assign(periodo=n).reset_index(names="variable") for n, m in out.items()]
    _guardar_csv(pd.concat(filas), DIR_FIG / "rezagos_diarios.csv")
    ic_tr = 1.96 / np.sqrt(max(int(es_train.sum()) // H_PRINCIPAL, 3))
    n_act = int((es_train & (f["fecha"] >= ini_actual).values).sum())
    ic_act = 1.96 / np.sqrt(max(n_act // H_PRINCIPAL, 3))
    print(f"\nLectura: L=0 es la variable de hoy; L=20 es su valor de hace 20 días hábiles. Ruido (95%): |ρ| < {ic_tr:.2f} en todo el train y |ρ| < {ic_act:.2f} en el régimen actual.")
    print("  Un desfase solo merece atención si supera ese umbral y mantiene el signo en ambos períodos; si no, es compatible con azar.")
    return out


def analisis_mensual(d: pd.DataFrame, ini_actual: pd.Timestamp) -> pd.DataFrame:
    """Hipótesis H1 y rezagos de ICG/ICC/inflación a nivel mensual (que es su frecuencia real de publicación)."""
    _titulo("6b) ICG / ICC / INFLACIÓN vs. variación del blue de los meses siguientes (hipótesis H1)")
    from scipy.stats import spearmanr
    x = d.assign(mes=d["fecha"].dt.to_period("M"))
    g = x.groupby("mes").agg(blue=("blue_venta", "last"), icg=("icg", "last"), icc=("icc", "last"),
                             infl=("inflacion_mensual_pct", "last"), fin=("fecha", "max"))
    ult = g.index[-1]
    if g.loc[ult, "fin"] < ult.to_timestamp(how="end").normalize() - pd.Timedelta(days=3):
        g = g.iloc[:-1]                                       # mes en curso incompleto: no se usa
    g["ret_mes"] = np.log(g["blue"]).diff() * 100
    ini_p = ini_actual.to_period("M")
    filas = []
    for var, etiqueta in (("icg", "ICG"), ("icc", "ICC"), ("infl", "Inflación (último dato publicado)")):
        for forma, serie in (("cambio mensual", g[var].diff()), ("nivel", g[var])):
            for L in (0, 1, 2, 3):
                yy = g["ret_mes"].shift(-L)
                for per, m in (("todo", np.ones(len(g), bool)), ("régimen actual", (g.index >= ini_p))):
                    ok = m & serie.notna().values & yy.notna().values
                    if ok.sum() >= 12:
                        rho, p = spearmanr(serie[ok], yy[ok])
                        filas.append({"variable": etiqueta, "forma": forma, "rezago_meses": L, "periodo": per,
                                      "spearman": rho, "p_valor": p, "n_meses": int(ok.sum())})
    r = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print("Rezago L = meses de adelanto: la variable del mes m contra la variación del blue en el mes m+L. L=0 es simultáneo; L≥1 es predictivo.")
    print(r[(r["rezago_meses"] == 1) & (r["forma"] == "cambio mensual") | (r["rezago_meses"].isin([1, 2]) & (r["forma"] == "nivel"))]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    h1 = r[(r["variable"] == "ICG") & (r["forma"] == "cambio mensual") & (r["rezago_meses"] == 1)]
    print("\nH1 (una caída del ICG anticipa una suba del blue a 15-30 días ⇒ correlación NEGATIVA entre ΔICG(m) y blue(m+1)):")
    for _, q in h1.iterrows():
        veredicto = "apoya H1 (signo negativo y p<0,05)" if (q["spearman"] < 0 and q["p_valor"] < 0.05) else "NO es concluyente (no hay evidencia estadística)"
        print(f"   {q['periodo']:<15} ρ = {q['spearman']:+.3f} | p = {q['p_valor']:.3f} | n = {q['n_meses']} meses → {veredicto}")
    print(f"  Cuidado: se probaron {len(r)} combinaciones; con p<0,05 se esperan ~{len(r)*0.05:.0f} falsos positivos por azar. Con ~{int(g['icg'].notna().sum())} meses la potencia es baja.")
    _guardar_csv(r, DIR_FIG / "rezagos_mensuales.csv")
    r.attrs["mensual"] = g
    return r


# ───────────────────────────── 7. ATÍPICOS ─────────────────────────────
def marcar_atipicos(f: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Isolation Forest ajustado SOLO con el entrenamiento. Marca (no borra) los días atípicos."""
    _titulo("7) DÍAS ATÍPICOS (Isolation Forest, ajustado con el entrenamiento)")
    f = f.copy()
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:  # noqa: BLE001
        print("  (scikit-learn no disponible:", e, ")")
        f["outlier_iforest"] = 0
        return f
    cols = [c for c in columnas if c != "regimen_brecha_alta"]
    tr = f["conjunto"] == "train"
    med, mu, sd = f.loc[tr, cols].median(), f.loc[tr, cols].mean(), f.loc[tr, cols].std().replace(0, 1)
    Z = ((f[cols].fillna(med) - mu) / sd).fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=CONTAMINACION_IFOREST, random_state=SEMILLA).fit(Z[tr])
    f["outlier_iforest"] = (iso.predict(Z) == -1).astype(int)
    f.loc[f["conjunto"] == "previo", "outlier_iforest"] = 0
    marc = f[(f["outlier_iforest"] == 1) & tr]
    meses = marc["fecha"].dt.to_period("M").value_counts().sort_index()
    print(f"Marcados en train: {len(marc)} de {int(tr.sum())} días ({len(marc)/max(int(tr.sum()),1)*100:.1f}%). Meses con más atípicos:",
          meses.sort_values(ascending=False).head(8).to_dict())
    print("  → se concentran en crisis reales (devaluaciones, PASO). No los elimines: usá la columna para analizar el error con y sin ellos,")
    print("    o entrená con y_winsor (objetivo recortado) para que 2-3 eventos no dominen el ajuste.")
    return f


# ───────────────────────────── 8. GRÁFICOS ─────────────────────────────
def hacer_graficos(d: pd.DataFrame, f: pd.DataFrame, rez: dict, mensual: pd.DataFrame, ranking: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("  (matplotlib no disponible, se omiten los gráficos:", e, ")")
        return
    DIR_FIG.mkdir(parents=True, exist_ok=True)

    # 1) Blue, oficial y brecha
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax[0].plot(d["fecha"], d["blue_venta"], label="Blue (venta)", lw=1.2)
    ax[0].plot(d["fecha"], d["dolar_oficial"], label="Oficial", lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Pesos por USD (escala log)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Dólar blue y oficial: cambios de escala y de régimen")
    ax[1].plot(d["fecha"], d["brecha_pct"], color="tab:red", lw=1)
    ax[1].axhline(UMBRAL_BRECHA_ALTA, ls="--", color="gray")
    ax[1].fill_between(d["fecha"], 0, d["brecha_pct"].where(d["brecha_pct"] > UMBRAL_BRECHA_ALTA), color="tab:red", alpha=.25)
    ax[1].set_ylabel("Brecha blue/oficial (%)"); ax[1].grid(alpha=.3)
    ax[1].set_title(f"Brecha (sombreado: régimen de cepo, brecha > {UMBRAL_BRECHA_ALTA:.0f}%)")
    fig.tight_layout(); fig.savefig(DIR_FIG / "01_blue_oficial_brecha.png", dpi=130); plt.close(fig)

    # 2) Distribución del objetivo por horizonte
    tr = f[f["conjunto"] == "train"]
    fig, ax = plt.subplots(figsize=(8, 5))
    datos = [tr[f"y_{h}"].dropna() for h in HORIZONTES]
    ax.boxplot(datos, tick_labels=[f"{h} días hábiles" for h in HORIZONTES], showfliers=True, flierprops={"markersize": 2, "alpha": .4})
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("Variación del blue a h días (%)"); ax.grid(alpha=.3)
    ax.set_title("Objetivo: variación futura del blue (train). Colas largas = devaluaciones reales")
    fig.tight_layout(); fig.savefig(DIR_FIG / "02_objetivo_por_horizonte.png", dpi=130); plt.close(fig)

    # 3) Desfases diarios
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for a_, (nombre, m) in zip(ax, rez.items()):
        im = a_.imshow(m.values, cmap="RdBu_r", vmin=-.3, vmax=.3, aspect="auto")
        a_.set_xticks(range(m.shape[1]), m.columns); a_.set_yticks(range(m.shape[0]), m.index)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if pd.notna(m.values[i, j]):
                    a_.text(j, i, f"{m.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        a_.set_title(f"{nombre}")
    fig.colorbar(im, ax=ax, shrink=.8, label="Spearman con y_h")
    fig.suptitle(f"Desfases: variable(t-L) vs. variación futura del blue a {H_PRINCIPAL} días hábiles")
    fig.savefig(DIR_FIG / "03_desfases_diarios.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # 4) Ranking
    top = ranking[(ranking["descartar"] == "") & (ranking["tipo"] == "dinámica")].copy()
    top = top.reindex(top["spearman_regimen_actual"].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    yy = np.arange(len(top))
    ax.barh(yy - .2, top["spearman_train"], .4, label="Todo el train")
    ax.barh(yy + .2, top["spearman_regimen_actual"], .4, label="Régimen actual")
    ax.set_yticks(yy, top["variable"]); ax.axvline(0, color="k", lw=.8); ax.legend(); ax.grid(alpha=.3)
    ax.set_xlabel(f"Spearman con y_{H_PRINCIPAL}")
    ax.axvspan(-ranking["ruido_actual_+-"].iloc[0], ranking["ruido_actual_+-"].iloc[0], color="gray", alpha=.15, label="ruido (régimen actual)")
    ax.legend(); ax.set_title("Variables de dinámica (orden: |ρ| en régimen actual). Sombreado = ruido")
    fig.tight_layout(); fig.savefig(DIR_FIG / "04_ranking_variables.png", dpi=130); plt.close(fig)

    # 5) H1: cambio mensual de ICG / ICC vs. variación del blue del mes siguiente
    g = mensual.attrs.get("mensual")
    if g is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a_, v in zip(ax, ("icg", "icc")):
            xx, yy2 = g[v].diff(), g["ret_mes"].shift(-1)
            ok = xx.notna() & yy2.notna()
            a_.scatter(xx[ok], yy2[ok], s=14, alpha=.6)
            a_.axhline(0, color="k", lw=.6); a_.axvline(0, color="k", lw=.6)
            a_.set_xlabel(f"Δ {v.upper()} del mes m"); a_.set_ylabel("Variación del blue en m+1 (%)"); a_.grid(alpha=.3)
            a_.set_title(f"{v.upper()}: ρ = {pd.Series(xx[ok]).corr(pd.Series(yy2[ok]), method='spearman'):+.2f} (n={int(ok.sum())})")
        fig.suptitle("H1: ¿una caída de la confianza anticipa una suba del blue?")
        fig.tight_layout(); fig.savefig(DIR_FIG / "05_h1_confianza_vs_blue.png", dpi=130); plt.close(fig)
    print(f"  Gráficos guardados en {DIR_FIG}/ (01 a 05).")


# ───────────────────────────── AYUDAS PARA EL MODELADO ─────────────────────────────
def winsorizar_objetivo(f: pd.DataFrame, h: int | None = None, p=(1, 99)) -> pd.Series:
    """Objetivo recortado a los percentiles p del TRAIN (para que 2-3 devaluaciones no dominen el ajuste). Evaluar siempre con el y sin recortar."""
    y = f[f"y_{h or H_PRINCIPAL}"]
    lo, hi = np.nanpercentile(y[f["conjunto"] == "train"], p)
    return y.clip(lo, hi)


def estandarizar(f: pd.DataFrame, columnas: list[str]):
    """Para modelos lineales/redes: (x - media_train) / desvío_train. Los árboles no la necesitan."""
    tr = f["conjunto"] == "train"
    mu, sd = f.loc[tr, columnas].mean(), f.loc[tr, columnas].std().replace(0, 1)
    return (f[columnas] - mu) / sd, mu, sd


def preparar_xy(f: pd.DataFrame, columnas: list[str], h: int | None = None, winsor: bool = False):
    """Devuelve X_train, y_train, X_test, y_test (test = holdout; usarlo UNA sola vez, al final)."""
    h = h or H_PRINCIPAL
    y = winsorizar_objetivo(f, h) if winsor else f[f"y_{h}"]
    tr = (f["conjunto"] == "train") & f[f"y_{h}"].notna()
    te = (f["conjunto"] == "test") & f[f"y_{h}"].notna()
    return f.loc[tr, columnas], y[tr], f.loc[te, columnas], f.loc[te, f"y_{h}"]


# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main(ruta=None, guardar: bool = True):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="An input array is constant")
    ruta = Path(ruta) if ruta else _buscar()
    print(f"analisis_eda_features v{VERSION} | dataset: {ruta}")
    d_all = _leer_csv(ruta)
    faltan = [c for c in COL_BASE if c not in d_all.columns]
    if faltan:
        raise ValueError(f"Al dataset le faltan columnas: {faltan}. ¿Corriste construir_dataset_dolar_blue.py v3.4?")
    for c in ("dolar_mep", "dolar_ccl"):
        if d_all[c].isna().all():
            raise ValueError(f"{c} está vacía: el modelo usa MEP y CCL; revisá la descarga en construir_dataset_dolar_blue.py.")
    d = d_all[d_all["fecha"].dt.dayofweek < 5].reset_index(drop=True)

    informe_calidad(d_all, d)

    _titulo("2) INGENIERÍA DE VARIABLES")
    f = construir_features(d)
    ok = verificar_sin_fuga(d)
    cols = columnas_features(f)
    print(f"{len(cols)} variables: {cols}")
    print(f"Objetivos: " + ", ".join(f"y_{h}" for h in HORIZONTES) + f" (variación % del blue a h días hábiles; principal: y_{H_PRINCIPAL}).")

    _titulo("3) CONJUNTOS DE ENTRENAMIENTO Y TEST")
    f = asignar_conjuntos(f)
    print(f"  El modelo usa MEP y CCL: el período de modelado empieza el {inicio_mep_ccl(f):%d/%m/%Y} (antes: 'previo').")
    print(f.groupby("conjunto").agg(desde=("fecha", "min"), hasta=("fecha", "max"), filas=("fecha", "size")).to_string())
    print("  previo = calentamiento de ventanas | purga = h días sin usar (el objetivo mira hacia adelante) | sin_objetivo = días recientes aún sin resultado.")
    tr_idx = np.flatnonzero((f["conjunto"] == "train").values)
    for i, (a_, b_) in enumerate(walk_forward(f), 1):
        print(f"  walk-forward {i}: entrena hasta {f.loc[a_[-1], 'fecha']:%d/%m/%Y} ({len(a_)} filas) → valida {f.loc[b_[0], 'fecha']:%d/%m/%Y} a {f.loc[b_[-1], 'fecha']:%d/%m/%Y}")
    del tr_idx

    base = evaluar_baselines(f)
    ini_actual = _inicio_regimen_actual(f)
    rank = ranking_variables(f, ini_actual)
    columnas_modelo = rank.loc[rank["descartar"] == "", "variable"].tolist()
    rez = analisis_rezagos_diarios(f, ini_actual)
    mensual = analisis_mensual(d, ini_actual)
    f = marcar_atipicos(f, columnas_modelo)
    f[f"y_{H_PRINCIPAL}_winsor"] = winsorizar_objetivo(f)

    _titulo("8) GRÁFICOS Y ARCHIVOS")
    hacer_graficos(d, f, rez, mensual, rank)
    if guardar:
        _guardar_csv(base, DIR_FIG / "baselines_test.csv")
        _guardar_csv(f, DIR_OUT / "features_dolar_blue.csv")
        print(f"  Guardado: {DIR_OUT / 'features_dolar_blue.csv'} ({len(f)} filas, {f.shape[1]} columnas) y tablas en {DIR_FIG}/")
    print(f"\nColumnas sugeridas para el modelo ({len(columnas_modelo)}): {columnas_modelo}")
    print("Siguiente paso: comparar_modelos.py (ElasticNet, Random Forest y SVR contra las líneas base de la sección 4).")
    info = {"columnas_modelo": columnas_modelo, "ranking": rank, "baselines": base, "rezagos_diarios": rez,
            "rezagos_mensuales": mensual, "inicio_regimen_actual": ini_actual, "sin_fuga": ok}
    return f, info


if __name__ == "__main__":
    main()
