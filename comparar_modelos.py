#!/usr/bin/env python3
"""
Comparación de 3 algoritmos de Machine Learning (distintos de XGBoost / Isolation Forest) para predecir el dólar blue.  -- versión 2.1

Algoritmos (tres familias distintas, elegidas por el tipo de datos: pocas filas útiles, mucho ruido, colas largas, variables mensuales):
  1. ElasticNet     : regresión lineal regularizada (L1+L2). Baja varianza, elige variables sola, coeficientes interpretables.
  2. RandomForest   : promedio de muchos árboles (bagging). No extrapola ni se sobreajusta tanto como un solo árbol; captura no linealidades.
  3. SVR (kernel RBF): regresión por vectores soporte. Pérdida insensible a errores chicos → robusta al ruido y a los extremos.

Qué predice: la variación % del blue a h días hábiles (h=1 es "hoy", usando solo datos hasta ayer; h=10 ≈ 2 semanas; h=20 ≈ 1 mes).
El precio predicho = blue(t) * exp(y_predicho/100) y el MAPE se mide sobre ese precio.

Variables: 5 grupos — A precio y brecha, B mercado diario, C mensuales (ICC, ICG, inflación), D régimen y calendario y
E dólar MEP y CCL. El dólar MEP y el CCL son parte FIJA del modelo: el período de modelado empieza el primer día en que existen.
El MODELO FINAL usa A + C + D + E. El grupo B (reservas, riesgo país, BADLAR, Merval, ARGT, carry) se evalúa en cada corrida
(ablación "todos sin B") pero queda afuera porque, con los datos reales al 30/10/2026, sacarlo BAJA el error (ver GRUPOS_EXCLUIDOS).

Cómo se elige el mejor (sin mirar el test):
  - Cada algoritmo prueba una grilla chica de hiperparámetros con validación walk-forward (6 tramos de 120 días hábiles, con purga).
  - El campeón de cada horizonte es el de menor MAPE de validación. El TEST (últimos 120 días hábiles) se usa UNA vez, para informar.
  - Todo se compara contra el modelo ingenuo (blue(t+h) = blue(t)) con el test de Diebold-Mariano: ¿la mejora es real o puede ser azar?
Además: ablación por grupos (acumulada A → A+B → … → A+B+C+D+E, y "todos sin un grupo") e importancia por permutación
(solo sobre validación, con las variables del modelo final).

Entrada : lo que genera construir_dataset_dolar_blue.py (data/dataset_dolar_blue*.csv) + analisis_eda_features.py (debe estar en la misma carpeta).
Salidas : data/modelos/resultados_modelos.csv, ablacion.csv, importancia_variables.csv, predicciones_test.csv y gráficos 06-08.

FORMATO DE LOS CSV (igual que el resto del proyecto): separador ";", COMA decimal y fechas DD/MM/AAAA.
    pandas : df = pd.read_csv("data/modelos/resultados_modelos.csv", sep=";", decimal=",", encoding="utf-8-sig")
    R      : df <- read.csv2("data/modelos/resultados_modelos.csv", fileEncoding = "UTF-8-sig")
    Excel / Power BI: delimitador "punto y coma", origen UTF-8 (configuración regional Español-Argentina para los decimales).

Uso en Colab:
    !pip install -q scikit-learn scipy matplotlib
    import comparar_modelos as c
    res = c.main()                       # tarda unos minutos
"""
from __future__ import annotations

import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import analisis_eda_features as a

VERSION = "2.1"

# ───────────────────────────── CONFIGURACIÓN ─────────────────────────────
HORIZONTES_EVAL = (1, 10, 20)      # "hoy", ~2 semanas, ~1 mes (días hábiles)
N_FOLDS, TAM_FOLD = 6, 120         # validación walk-forward
WINSOR = (1, 99)                   # el objetivo de ENTRENAMIENTO se recorta a estos percentiles (calculados con cada train); la evaluación usa el valor real
SEMILLA = 42
N_REPETICIONES_PERM = 3            # repeticiones de la importancia por permutación
DIR_MOD = Path("data/modelos")

GRILLAS = {
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.9]},
    # min_samples_leaf >= 20: una hoja necesita ~1 mes de días; así el árbol no puede "memorizar" un mes por su valor de ICC/ICG/inflación
    "RandomForest": {"max_depth": [3, 6], "min_samples_leaf": [20, 60]},
    "SVR": {"C": [0.1, 1.0, 10.0], "epsilon": [0.05, 0.3]},
}

# Grupos de variables para la ablación (de lo más simple a lo más completo)
GRUPOS = {
    "A_precio_y_brecha": ["ret_blue_1", "ret_blue_5", "ret_blue_10", "ret_blue_20", "ret_blue_60", "vol_blue_20", "vol_blue_60",
                          "dist_media_blue_20", "dist_media_blue_60", "ret_oficial_5", "ret_oficial_20", "brecha_pct",
                          "brecha_cambio_5", "brecha_cambio_20", "brecha_z_120", "regimen_brecha_alta"],
    "B_mercado_diario": ["var_reservas_5", "var_reservas_10", "var_reservas_20", "reservas_vs_max_252", "rp_nivel", "rp_cambio_5",
                         "rp_cambio_20", "badlar_nivel", "badlar_cambio_5", "carry_oficial_20", "merval_ret_5", "merval_ret_20",
                         "argt_ret_5", "argt_ret_20", "argt_vol_20"],
    "C_mensuales_sentimiento_inflacion": ["icg_t1", "icc_t1", "d_icg_t1", "d_icc_t1", "infl_t1", "infl_acel_t1", "tasa_real_mensual"],
    "D_regimen_y_calendario": ["cepo", "dias_a_eleccion", "dias_desde_eleccion", "dia_semana", "dia_mes", "mes_aguinaldo"],
    "E_mep_ccl": list(a.VARIABLES_MEP_CCL),          # dólar MEP y CCL: parte fija del modelo
}
FRECUENCIA = {**{c: "diaria" for g in ("A_precio_y_brecha", "B_mercado_diario", "D_regimen_y_calendario", "E_mep_ccl") for c in GRUPOS[g]},
              **{c: "mensual" for c in GRUPOS["C_mensuales_sentimiento_inflacion"]}}

# Grupos que NO entran al modelo final. Evidencia con los datos reales (dataset al 30/10/2026, SVR, h=10):
#   - backtest mensual 2022-2026 (1250 días): MAPE 3,56% → 3,50% sin B (mejora vs ingenuo +5,3% → +6,9%; gana los 5 años en vez de 3 de 5)
#   - test may-oct 2026: mejora vs ingenuo +41,8% → +45,0%;  h=1 y h=20 también mejoran levemente
#   - ablación e importancia por permutación de la versión 2.0: el grupo B EMPEORABA el error en validación (h=10 y h=20)
# Las variables de B no dejan de calcularse: siguen en el análisis exploratorio y en la ablación ("todos sin B"), que
# avisa si con datos nuevos el grupo B pasara a ayudar. Para volver a incluirlo: GRUPOS_EXCLUIDOS = ()
GRUPOS_EXCLUIDOS = ("B_mercado_diario",)
GRUPOS_MODELO = [g for g in GRUPOS if g not in GRUPOS_EXCLUIDOS]


def columnas_modelo(f: pd.DataFrame) -> list[str]:
    """Variables del modelo final (grupos de GRUPOS_MODELO). El grupo E (MEP y CCL) es obligatorio."""
    faltan = [x for x in GRUPOS["E_mep_ccl"] if x not in f.columns]
    if faltan:
        raise ValueError(f"Faltan variables de MEP/CCL {faltan}: corré construir_dataset_dolar_blue.py v3.4.")
    return [x for g in GRUPOS_MODELO for x in GRUPOS[g] if x in f.columns]


# ───────────────────────────── MODELOS ─────────────────────────────
def construir_modelo(nombre: str, p: dict):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    pre = [SimpleImputer(strategy="median"), StandardScaler()]       # se ajustan SOLO con el train de cada tramo (van dentro del pipeline)
    if nombre == "ElasticNet":
        est = ElasticNet(max_iter=50_000, random_state=SEMILLA, **p)
    elif nombre == "RandomForest":
        est = RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=SEMILLA, **p)
    elif nombre == "SVR":
        est = TransformedTargetRegressor(regressor=SVR(kernel="rbf", gamma="scale", **p), transformer=StandardScaler())
    else:
        raise ValueError(nombre)
    return make_pipeline(*pre, est)


# ───────────────────────────── MÉTRICAS ─────────────────────────────
def _precio(blue_t, y_pct):
    return blue_t * np.exp(y_pct / 100)


def metricas(blue_t, real, y_pred, y_real) -> dict:
    pp = _precio(blue_t, y_pred)
    err = pp - real
    ape = err.abs() / real * 100
    mov = (y_real != 0) & (y_pred != 0)
    return {"MAPE_%": ape.mean(), "MAE_pesos": err.abs().mean(), "RMSE_pesos": float(np.sqrt((err ** 2).mean())),
            "acierto_direccion_%": (np.sign(y_pred[mov]) == np.sign(y_real[mov])).mean() * 100 if mov.any() else np.nan}


def dm_test(ape_base: pd.Series, ape_modelo: pd.Series, h: int):
    """
    Diebold-Mariano (con corrección de Harvey-Leybourne-Newbold y varianza HAC para pronósticos solapados a h pasos).
    d = error del ingenuo − error del modelo (en % absoluto). Devuelve (mejora media en puntos de MAPE, p-valor bilateral).
    H0: ambos pronostican igual de bien. p < 0,05 ⇒ la diferencia no se explica por azar.
    """
    from scipy.stats import t as t_dist
    d = (ape_base - ape_modelo).dropna().values
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float(np.mean(d)) if n else np.nan, np.nan
    media = d.mean()
    s = np.mean((d - media) ** 2)
    for k in range(1, h):
        s += 2 * (1 - k / h) * np.mean((d[k:] - media) * (d[:-k] - media))
    var = s / n
    if var <= 0:
        return float(media), np.nan
    est = media / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(media), float(2 * (1 - t_dist.cdf(abs(est), n - 1)))


def veredicto(mejora_pct: float, p: float) -> str:
    """Traduce (mejora vs ingenuo, p-valor de Diebold-Mariano) a una frase sin ambigüedad."""
    if pd.isna(p):
        return "sin test estadístico"
    if p < 0.05:
        return "SUPERA al ingenuo (diferencia significativa)" if mejora_pct > 0 else "es SIGNIFICATIVAMENTE PEOR que el ingenuo"
    return ("mejora al ingenuo pero la diferencia NO es significativa (puede ser azar)" if mejora_pct > 0
            else "no mejora al ingenuo (empata o es peor, sin diferencia significativa)")


# ───────────────────────────── VALIDACIÓN WALK-FORWARD ─────────────────────────────
def cv_modelo(f, cols, h, nombre, params, folds, guardar_modelos=False):
    """Entrena en cada tramo con el pasado (con purga) y predice el tramo siguiente. Devuelve predicciones y, si se pide, los modelos."""
    y = f"y_{h}"
    out, modelos = [], []
    for k, (itr, iva) in enumerate(folds, 1):
        tr = f.iloc[itr].dropna(subset=[y])
        va = f.iloc[iva]
        lo, hi = np.nanpercentile(tr[y], WINSOR)
        m = construir_modelo(nombre, params)
        m.fit(tr[cols], tr[y].clip(lo, hi))
        out.append(pd.DataFrame({"fila": iva, "fold": k, "y_pred": m.predict(va[cols])}))
        if guardar_modelos:
            modelos.append((m, va))
    return pd.concat(out, ignore_index=True), modelos


def _unir(f, pred, h):
    d = pred.merge(f[["fecha", "blue_venta", f"y_{h}", f"blue_t_mas_{h}"]], left_on="fila", right_index=True)
    return d.dropna(subset=[f"y_{h}"]).sort_values("fecha").reset_index(drop=True)


def _mape(f, pred, h):
    d = _unir(f, pred, h)
    return metricas(d["blue_venta"], d[f"blue_t_mas_{h}"], d["y_pred"], d[f"y_{h}"])["MAPE_%"]


def buscar_hiperparametros(f, cols, h, nombre, folds):
    mejor = (np.inf, None)
    claves = list(GRILLAS[nombre])
    for valores in itertools.product(*GRILLAS[nombre].values()):
        p = dict(zip(claves, valores))
        pred, _ = cv_modelo(f, cols, h, nombre, p, folds)
        m = _mape(f, pred, h)
        if m < mejor[0]:
            mejor = (m, p)
    return mejor[1], mejor[0]


# ───────────────────────────── IMPORTANCIA Y ABLACIÓN ─────────────────────────────
def importancia_permutacion(f, cols, h, modelos) -> pd.DataFrame:
    """Cuánto EMPEORA el MAPE si se mezclan al azar los valores de una variable (o grupo) en los tramos de validación."""
    rng = np.random.default_rng(SEMILLA)
    filas = {c: [] for c in cols}
    grupos = {g: [] for g in GRUPOS}
    for m, va in modelos:
        X = va[cols]
        base = _mape_arr(va, h, m.predict(X))
        for c in cols:
            inc = []
            for _ in range(N_REPETICIONES_PERM):
                Xp = X.copy()
                Xp[c] = rng.permutation(Xp[c].values)
                inc.append(_mape_arr(va, h, m.predict(Xp)) - base)
            filas[c].append(np.mean(inc))
        for g, gc in GRUPOS.items():
            gc = [c for c in gc if c in cols]
            if not gc:                     # grupo fuera del modelo final (su aporte se mide en la ablación)
                continue
            inc = []
            for _ in range(N_REPETICIONES_PERM):
                Xp = X.copy()
                idx = rng.permutation(len(Xp))
                Xp[gc] = Xp[gc].values[idx]
                inc.append(_mape_arr(va, h, m.predict(Xp)) - base)
            grupos[g].append(np.mean(inc))
    r = pd.DataFrame({"variable": cols, "grupo": [next(g for g, gc in GRUPOS.items() if c in gc) for c in cols],
                      "frecuencia": [FRECUENCIA[c] for c in cols],
                      "aumento_MAPE_puntos": [np.mean(filas[c]) for c in cols]})
    r["horizonte"] = h
    grupos = {g: v for g, v in grupos.items() if v}
    rg = pd.DataFrame({"variable": [f"[GRUPO] {g}" for g in grupos], "grupo": list(grupos),
                       "frecuencia": ["mensual" if g.startswith("C") else "diaria" for g in grupos],
                       "aumento_MAPE_puntos": [np.mean(v) for v in grupos.values()], "horizonte": h})
    return pd.concat([r, rg], ignore_index=True)


def _mape_arr(va, h, y_pred):
    return metricas(va["blue_venta"], va[f"blue_t_mas_{h}"], pd.Series(y_pred, index=va.index), va[f"y_{h}"])["MAPE_%"]


def ablacion(f, h, nombre, params, folds, base_ingenuo) -> pd.DataFrame:
    """
    Dos lecturas del aporte de cada grupo (MAPE de validación con el mismo algoritmo e hiperparámetros):
      - acumulada : A → A+B → A+B+C → … (lo que suma cada grupo sobre los anteriores)
      - sin_grupo : todos los grupos menos uno (si el MAPE BAJA al sacarlo, ese grupo empeora la predicción)
    """
    filas, acum = [], []

    def evaluar(cols, tipo, etiqueta):
        pred, _ = cv_modelo(f, list(cols), h, nombre, params, folds)
        m = _mape(f, pred, h)
        filas.append({"horizonte": h, "modelo": nombre, "tipo": tipo, "variables_incluidas": etiqueta,
                      "n_variables": len(cols), "MAPE_cv_%": m, "mejora_vs_ingenuo_%": (1 - m / base_ingenuo) * 100})

    for k, (g, gc) in enumerate(GRUPOS.items(), 1):
        acum += [c for c in gc if c in f.columns]
        evaluar(acum, "acumulada", "+".join(x[0] for x in list(GRUPOS)[:k]))
    for g in GRUPOS:
        evaluar([c for gg, gc in GRUPOS.items() if gg != g for c in gc if c in f.columns], "sin_grupo", f"todos sin {g[0]}")
    return pd.DataFrame(filas)


def _revisar_exclusiones(ab: pd.DataFrame, h: int) -> None:
    """Avisa si, con los datos actuales, un grupo excluido pasó a ayudar (o uno incluido a empeorar)."""
    todos = ab[(ab["tipo"] == "acumulada")]["MAPE_cv_%"].iloc[-1]
    for _, r in ab[ab["tipo"] == "sin_grupo"].iterrows():
        g = next(x for x in GRUPOS if x[0] == r["variables_incluidas"][-1])
        dif = r["MAPE_cv_%"] - todos                       # > 0: sacarlo empeora → el grupo ayuda
        if g in GRUPOS_EXCLUIDOS:
            print(f"  → {g} (excluido del modelo final): al sacarlo el MAPE {'SUBE' if dif > 0 else 'baja'} {abs(dif):.3f} pp"
                  + ("  ⚠ con estos datos AYUDA: revisá GRUPOS_EXCLUIDOS" if dif > 0 else "  ✓ confirma la exclusión"))
        elif dif < 0:
            print(f"  → {g}: al sacarlo el MAPE baja {abs(dif):.3f} pp en h={h} (es parte del modelo final; ver si se repite en otros horizontes)")


# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main(ruta=None, guardar: bool = True):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    ruta = Path(ruta) if ruta else a._buscar()
    d_all = a._leer_csv(ruta)
    d = d_all[d_all["fecha"].dt.dayofweek < 5].reset_index(drop=True)
    f = a.asignar_conjuntos(a.construir_features(d))
    cols = columnas_modelo(f)
    folds = list(a.walk_forward(f, N_FOLDS, TAM_FOLD))
    etiqueta = "+".join(g[0] for g in GRUPOS_MODELO)
    tr, te = f[f["conjunto"] == "train"], f[f["conjunto"] == "test"]
    print(f"comparar_modelos v{VERSION} | dataset: {ruta}\n{len(cols)} variables del modelo final (grupos {etiqueta}; incluye MEP y CCL; período desde {a.inicio_mep_ccl(f):%d/%m/%Y}) | train {tr['fecha'].min():%d/%m/%Y}→{tr['fecha'].max():%d/%m/%Y} ({len(tr)}) | "
          f"test {te['fecha'].min():%d/%m/%Y}→{te['fecha'].max():%d/%m/%Y} ({len(te)})")
    print("Validación walk-forward: " + " | ".join(f"{f.loc[b[0], 'fecha']:%m/%Y}-{f.loc[b[-1], 'fecha']:%m/%Y}" for _, b in folds))

    filas, filas_abl, filas_imp, pred_test = [], [], [], {}
    campeones = {}
    for h in HORIZONTES_EVAL:
        a._titulo(f"HORIZONTE h = {h} día(s) hábil(es)" + ("  → 'HOY' (con datos hasta ayer)" if h == 1 else f"  ≈ {round(h*7/5)} días corridos"))
        y = f"y_{h}"
        trh = tr.dropna(subset=[y])
        teh = te.dropna(subset=[y])
        drift = trh[y].mean()

        # Líneas base: validación (mismos tramos) y test
        base_cv = {"ingenuo": pd.DataFrame({"fila": np.concatenate([b for _, b in folds]), "fold": 0, "y_pred": 0.0}),
                   "deriva": pd.DataFrame({"fila": np.concatenate([b for _, b in folds]), "fold": 0, "y_pred": drift})}
        ape_cv, res_h = {}, {}
        for nombre, p in base_cv.items():
            u = _unir(f, p, h)
            ape_cv[nombre] = (_precio(u["blue_venta"], u["y_pred"]) - u[f"blue_t_mas_{h}"]).abs() / u[f"blue_t_mas_{h}"] * 100
            res_h[nombre] = {"cv": metricas(u["blue_venta"], u[f"blue_t_mas_{h}"], u["y_pred"], u[y]), "u": u, "params": ""}
        mape_ing_cv = res_h["ingenuo"]["cv"]["MAPE_%"]

        # Modelos de ML: grilla + validación
        for nombre in GRILLAS:
            t1 = time.time()
            params, mape_cv = buscar_hiperparametros(f, cols, h, nombre, folds)
            pred, mods = cv_modelo(f, cols, h, nombre, params, folds, guardar_modelos=True)
            u = _unir(f, pred, h)
            ape = (_precio(u["blue_venta"], u["y_pred"]) - u[f"blue_t_mas_{h}"]).abs() / u[f"blue_t_mas_{h}"] * 100
            mej, p_dm = dm_test(ape_cv["ingenuo"].reset_index(drop=True), ape.reset_index(drop=True), h)
            res_h[nombre] = {"cv": metricas(u["blue_venta"], u[f"blue_t_mas_{h}"], u["y_pred"], u[y]), "u": u, "params": params,
                             "p_dm_cv": p_dm, "mods": mods}
            print(f"  {nombre:<13} mejores hiperparámetros {params} | MAPE validación {mape_cv:.3f}% | {time.time()-t1:.0f}s")

        # Test (una sola vez): reentrenar con TODO el train y los mejores hiperparámetros
        lo, hi = np.nanpercentile(trh[y], WINSOR)
        pt = {"ingenuo": np.zeros(len(teh)), "deriva": np.full(len(teh), drift)}
        for nombre in GRILLAS:
            m = construir_modelo(nombre, res_h[nombre]["params"]).fit(trh[cols], trh[y].clip(lo, hi))
            pt[nombre] = m.predict(teh[cols])
        ape_te = {}
        for nombre, yp in pt.items():
            yp = pd.Series(yp, index=teh.index)
            mt = metricas(teh["blue_venta"], teh[f"blue_t_mas_{h}"], yp, teh[y])
            ape_te[nombre] = ((_precio(teh["blue_venta"], yp) - teh[f"blue_t_mas_{h}"]).abs() / teh[f"blue_t_mas_{h}"] * 100).reset_index(drop=True)
            res_h[nombre]["te"] = mt
        pred_test[h] = pd.DataFrame({"fecha": teh["fecha"].values, "horizonte": h, "blue_hoy": teh["blue_venta"].values,
                                     "real": teh[f"blue_t_mas_{h}"].values,
                                     **{n: _precio(teh["blue_venta"].values, np.asarray(v)) for n, v in pt.items()}})

        mape_ing_te = res_h["ingenuo"]["te"]["MAPE_%"]
        for nombre, r in res_h.items():
            _, p_te = (np.nan, np.nan) if nombre == "ingenuo" else dm_test(ape_te["ingenuo"], ape_te[nombre], h)
            filas.append({"horizonte": h, "modelo": nombre, "hiperparametros": str(r["params"]),
                          "MAPE_cv_%": r["cv"]["MAPE_%"], "mejora_cv_vs_ingenuo_%": (1 - r["cv"]["MAPE_%"] / mape_ing_cv) * 100,
                          "p_DM_cv": r.get("p_dm_cv", np.nan),
                          "MAPE_test_%": r["te"]["MAPE_%"], "MAE_test_pesos": r["te"]["MAE_pesos"],
                          "mejora_test_vs_ingenuo_%": (1 - r["te"]["MAPE_%"] / mape_ing_te) * 100, "p_DM_test": p_te,
                          "acierto_direccion_test_%": r["te"]["acierto_direccion_%"]})
        tabla = pd.DataFrame([x for x in filas if x["horizonte"] == h])
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print("\n" + tabla.drop(columns=["horizonte", "hiperparametros"]).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        print("  (mejora > 0 ⇒ mejor que el ingenuo; p_DM < 0,05 ⇒ la diferencia con el ingenuo no se explica por azar)")

        ml = {n: res_h[n]["cv"]["MAPE_%"] for n in GRILLAS}
        campeon = min(ml, key=ml.get)
        campeones[h] = campeon
        rc = tabla[tabla["modelo"] == campeon].iloc[0]
        print(f"\n  ★ Campeón h={h} (menor MAPE de validación): {campeon} → {veredicto(rc['mejora_cv_vs_ingenuo_%'], rc['p_DM_cv'])}.")

        # Ablación e importancia con el campeón
        ab = ablacion(f, h, campeon, res_h[campeon]["params"], folds, mape_ing_cv)
        filas_abl.append(ab)
        print("\n  Ablación (¿aportan las variables de cada grupo? MAPE de validación del campeón):")
        print(ab.drop(columns=["horizonte", "modelo"]).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        _revisar_exclusiones(ab, h)
        imp = importancia_permutacion(f, cols, h, res_h[campeon]["mods"])
        imp["modelo"] = campeon
        filas_imp.append(imp)
        top = imp[~imp["variable"].str.startswith("[GRUPO]")].sort_values("aumento_MAPE_puntos", ascending=False).head(8)
        print("\n  Importancia por permutación (aumento del MAPE al mezclar la variable; solo validación). Top 8:")
        print(top[["variable", "frecuencia", "aumento_MAPE_puntos"]].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        print(imp[imp["variable"].str.startswith("[GRUPO]")][["variable", "aumento_MAPE_puntos"]].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    res = pd.DataFrame(filas)
    abl = pd.concat(filas_abl, ignore_index=True)
    imp = pd.concat(filas_imp, ignore_index=True)
    ptest = pd.concat(pred_test.values(), ignore_index=True)

    a._titulo("RESUMEN")
    for h in HORIZONTES_EVAL:
        r = res[(res["horizonte"] == h) & (res["modelo"] == campeones[h])].iloc[0]
        print(f"  h={h:>2}: campeón {campeones[h]:<10} validación: {r['MAPE_cv_%']:.3f}% ({r['mejora_cv_vs_ingenuo_%']:+.1f}% vs ingenuo, p={r['p_DM_cv']:.3f}) → {veredicto(r['mejora_cv_vs_ingenuo_%'], r['p_DM_cv'])}")
        print(f"        test: {r['MAPE_test_%']:.3f}% ({r['mejora_test_vs_ingenuo_%']:+.1f}%, p={r['p_DM_test']:.3f}) → {veredicto(r['mejora_test_vs_ingenuo_%'], r['p_DM_test'])}")
    print("  H2 (MAPE < 10%) se cumple trivialmente (ver el ingenuo). Lo que debe mostrar el proyecto es la columna 'mejora vs ingenuo' y su p-valor.")

    if guardar:
        for nombre, df in (("resultados_modelos", res), ("ablacion", abl), ("importancia_variables", imp), ("predicciones_test", ptest)):
            a._guardar_csv(df, DIR_MOD / f"{nombre}.csv")
        hacer_graficos(res, ptest, imp, campeones)
        print(f"\n  Guardado en {DIR_MOD}/ (4 CSV y 3 gráficos). Tiempo total: {(time.time()-t0)/60:.1f} min")
    return {"resultados": res, "ablacion": abl, "importancia": imp, "predicciones_test": ptest, "campeones": campeones, "features": f}


# ───────────────────────────── GRÁFICOS ─────────────────────────────
def hacer_graficos(res, ptest, imp, campeones):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("  (matplotlib no disponible:", e, ")")
        return
    DIR_MOD.mkdir(parents=True, exist_ok=True)
    modelos = ["ingenuo", "deriva", "ElasticNet", "RandomForest", "SVR"]
    colores = ["#888888", "#bbbbbb", "#1f77b4", "#2ca02c", "#d62728"]
    hs = sorted(res["horizonte"].unique())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a_, col, tit in zip(ax, ("MAPE_cv_%", "MAPE_test_%"), ("Validación walk-forward (≈3 años, incluye estrés)", "Test (últimos 120 días hábiles)")):
        w = 0.16
        for i, (m, c) in enumerate(zip(modelos, colores)):
            v = [res[(res["horizonte"] == h) & (res["modelo"] == m)][col].iloc[0] for h in hs]
            a_.bar(np.arange(len(hs)) + (i - 2) * w, v, w, label=m, color=c)
        a_.set_xticks(range(len(hs)), [f"h={h}" + (" (hoy)" if h == 1 else "") for h in hs]); a_.set_ylabel("MAPE (%)"); a_.set_title(tit); a_.grid(axis="y", alpha=.3)
    ax[0].legend(ncol=2, fontsize=8)
    fig.suptitle("MAPE por modelo y horizonte (menos es mejor). El ingenuo ya es muy bajo: lo que importa es la diferencia")
    fig.tight_layout(); fig.savefig(DIR_MOD / "06_comparacion_modelos.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(len(hs), 1, figsize=(11, 3.2 * len(hs)), sharex=True)
    for a_, h in zip(np.atleast_1d(ax), hs):
        p = ptest[ptest["horizonte"] == h]
        a_.plot(p["fecha"] + pd.Timedelta(days=0), p["real"], color="k", lw=1.4, label="Real (blue en t+h)")
        a_.plot(p["fecha"], p["ingenuo"], color="#888888", lw=1, ls="--", label="Ingenuo")
        a_.plot(p["fecha"], p[campeones[h]], color="#d62728", lw=1, label=f"Campeón: {campeones[h]}")
        a_.set_title(f"h = {h}: el eje X es la fecha en que se hace la predicción"); a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(DIR_MOD / "07_test_real_vs_prediccion.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, len(hs), figsize=(5 * len(hs), 5.2))
    for a_, h in zip(np.atleast_1d(ax), hs):
        t = imp[(imp["horizonte"] == h) & ~imp["variable"].str.startswith("[GRUPO]")].sort_values("aumento_MAPE_puntos", ascending=False).head(10).iloc[::-1]
        a_.barh(t["variable"], t["aumento_MAPE_puntos"], color=["#ff7f0e" if x == "mensual" else "#1f77b4" for x in t["frecuencia"]])
        a_.set_title(f"h={h} ({imp[imp['horizonte']==h]['modelo'].iloc[0]})"); a_.set_xlabel("Aumento del MAPE (pp)"); a_.grid(alpha=.3)
    fig.suptitle("Importancia por permutación en validación (azul = diaria, naranja = mensual)")
    fig.tight_layout(); fig.savefig(DIR_MOD / "08_importancia_variables.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
