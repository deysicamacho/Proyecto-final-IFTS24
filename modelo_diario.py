#!/usr/bin/env python3
"""
Modelo diario del dólar blue: predice el valor de CADA DÍA con los datos disponibles hasta el día anterior.  -- versión 2.1

Qué hace (como funcionaría la herramienta en producción, día por día):
  1. Simulación histórica ("backtest" con origen móvil): al comienzo de cada mes entrena el modelo SOLO con el pasado y,
     durante ese mes, cada día predice el blue del día hábil siguiente. Así se obtiene una predicción para todos los días
     desde INICIO_BACKTEST, comparable con el valor real, sin haber usado nunca información futura.
  2. El modelo (por defecto SVR, el mejor en la validación; o un ensamble de varios) usa las variables del modelo final de
     comparar_modelos.py (grupos A, C, D y E; SIEMPRE incluye el dólar MEP y CCL) y se entrena solo con los días en que
     existen MEP y CCL. Se entrena sobre la VARIACIÓN del blue:
         blue_predicho(hoy) = blue(ayer) × exp(variación_predicha / 100)
     Es Machine Learning prediciendo el VALOR del blue: el valor de ayer es la base y el modelo predice cuánto se mueve.
     (Predecir el nivel directo con árboles falla: no pueden dar valores mayores a los vistos y el blue sube por inflación.)
  3. Rango probable (intervalo conformal adaptativo): con los errores de los últimos VENTANA_ERRORES días ya conocidos,
     arma un rango que contiene el valor real el 80% y el 95% de las veces. Se informa la cobertura real obtenida.
  4. Alerta de incertidumbre (Isolation Forest, como en el documento): si el día es atípico respecto del entrenamiento,
     se marca "alta incertidumbre" y se recomienda usar el rango del 95%.
  5. Predicción del próximo día hábil, entrenando con todos los datos.

Entrada : data/dataset_dolar_blue*.csv  + analisis_eda_features.py y comparar_modelos.py en la misma carpeta.
Salidas : data/modelo/predicciones_diarias.csv   (una fila por día: real, predicho, rango 80/95, alerta, error, ingenuo)
          data/modelo/metricas_backtest.csv      (MAPE por año y por régimen, contra el ingenuo, cobertura del rango)
          data/modelo/prediccion_proximo_dia.csv (lo que mostraría la app)
          data/modelo/09_real_vs_predicho.png, 10_error_mensual.png

FORMATO DE LOS CSV (igual que el resto del proyecto): separador ";", COMA decimal y fechas DD/MM/AAAA.
    pandas : df = pd.read_csv("data/modelo/predicciones_diarias.csv", sep=";", decimal=",", encoding="utf-8-sig")
             df["fecha_predicha"] = pd.to_datetime(df["fecha_predicha"], format="%d/%m/%Y")
    R      : df <- read.csv2("data/modelo/predicciones_diarias.csv", fileEncoding = "UTF-8-sig")
    Excel / Power BI: delimitador "punto y coma", origen UTF-8 (configuración regional Español-Argentina para los decimales).

Uso en Colab:
    !pip install -q scikit-learn scipy matplotlib
    import modelo_diario as md
    res = md.main()                 # h=1: el blue de "hoy" con datos hasta ayer (≈ 3-6 minutos)
    res = md.main(h=10)             # también sirve para ~2 semanas
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import analisis_eda_features as a
import comparar_modelos as c

VERSION = "2.1"

# ───────────────────────────── CONFIGURACIÓN ─────────────────────────────
INICIO_BACKTEST = "2022-01-01"     # primer mes que se predice día por día (antes: solo entrenamiento)
# SVR solo fue el mejor en la validación walk-forward; el promedio de los 3 no lo mejoró. Para probar el ensamble:
#   MODELOS_ENSAMBLE = ("ElasticNet", "RandomForest", "SVR")
MODELOS_ENSAMBLE = ("SVR",)
# Hiperparámetros por horizonte. Para el SVR se revisaron con los datos reales (backtest mensual 2022-abr 2026, sin mirar el test):
# C=0.03 o 0.3, epsilon=0.3, gamma menor, ventana de 2-3 años, pesos a lo reciente, objetivo recortado al 5/95 y ensambles
# SVR+RF o SVR+ElasticNet NO mejoraron de forma consistente a C=0.1, epsilon=0.05 en ningún horizonte. ElasticNet y RandomForest
# solo se usan si se arma un ensamble (MODELOS_ENSAMBLE).
HIPERPARAMETROS = {
    1: {"ElasticNet": {"alpha": 0.1, "l1_ratio": 0.2}, "RandomForest": {"max_depth": 6, "min_samples_leaf": 60}, "SVR": {"C": 0.1, "epsilon": 0.05}},
    10: {"ElasticNet": {"alpha": 1.0, "l1_ratio": 0.5}, "RandomForest": {"max_depth": 3, "min_samples_leaf": 60}, "SVR": {"C": 0.1, "epsilon": 0.05}},
    20: {"ElasticNet": {"alpha": 1.0, "l1_ratio": 0.2}, "RandomForest": {"max_depth": 6, "min_samples_leaf": 20}, "SVR": {"C": 0.1, "epsilon": 0.05}},
}
VIDA_MEDIA_DIAS = None             # peso a lo reciente: None = todos los días pesan igual (730 = un dato de hace 2 años pesa la mitad; se probó y empeoró)
VENTANA_ERRORES = 120              # días de errores pasados para armar el rango probable
COBERTURAS = (0.80, 0.95)
CONTAMINACION_IFOREST = 0.02
SEMILLA = 42
DIR_MODELO = Path("data/modelo")


# ───────────────────────────── ENTRENAMIENTO ─────────────────────────────
def _columnas(f: pd.DataFrame) -> list[str]:
    """Variables del modelo final (mismas que comparar_modelos: grupos A, C, D y E; el grupo E, MEP y CCL, es obligatorio)."""
    return c.columnas_modelo(f)


def _con_datos(tr: pd.DataFrame) -> pd.DataFrame:
    """Solo días con MEP y CCL: el modelo no se entrena con años en que esos datos no existían (no se rellenan)."""
    return tr.dropna(subset=["brecha_blue_mep_pct", "brecha_blue_ccl_pct"])


def entrenar(tr: pd.DataFrame, cols: list[str], h: int):
    """Entrena los modelos del ensamble con las filas `tr` (objetivo recortado a los percentiles del propio train)."""
    y = f"y_{h}"
    tr = tr.dropna(subset=[y])
    lo, hi = np.nanpercentile(tr[y], c.WINSOR)
    modelos = []
    for nombre in MODELOS_ENSAMBLE:
        m = c.construir_modelo(nombre, HIPERPARAMETROS[h][nombre])
        kw = {}
        if VIDA_MEDIA_DIAS:
            edad = (tr["fecha"].max() - tr["fecha"]).dt.days.values
            kw = {f"{m.steps[-1][0]}__sample_weight": 0.5 ** (edad / VIDA_MEDIA_DIAS)}
        m.fit(tr[cols], tr[y].clip(lo, hi), **kw)
        modelos.append(m)
    return modelos


def predecir(modelos, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict(X) for m in modelos], axis=0)


def _detector_atipicos(tr: pd.DataFrame, cols: list[str]):
    from sklearn.ensemble import IsolationForest
    med, mu, sd = tr[cols].median(), tr[cols].mean(), tr[cols].std().replace(0, 1)
    z = lambda X: ((X[cols].fillna(med) - mu) / sd).fillna(0)
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINACION_IFOREST, random_state=SEMILLA).fit(z(tr))
    return lambda X: (iso.predict(z(X)) == -1).astype(int)


# ───────────────────────────── BACKTEST DÍA POR DÍA ─────────────────────────────
def backtest(f: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Para cada mes: entrena con lo conocido al inicio del mes y predice cada día de ese mes. Nunca usa datos futuros."""
    cols = _columnas(f)
    y = f"y_{h}"
    inicio_modelo = pd.Timestamp(a.FECHA_INICIO_MODELO)
    salida = []
    meses = pd.period_range(INICIO_BACKTEST, f["fecha"].max(), freq="M")
    for k, mes in enumerate(meses, 1):
        pr = f[(f["fecha"] >= mes.start_time) & (f["fecha"] <= mes.end_time)]
        if pr.empty:
            continue
        i0 = pr.index[0]
        # Filas de entrenamiento: su resultado (blue en t+h) ya se conocía el día i0 → posición <= i0 - h
        tr = f.loc[: i0 - h]
        tr = _con_datos(tr[(tr["fecha"] >= inicio_modelo)].dropna(subset=[y]))
        if len(tr) < 250:                                   # muy pocos días con MEP/CCL para entrenar
            continue
        modelos = entrenar(tr, cols, h)
        atipico = _detector_atipicos(tr, cols)
        yp = predecir(modelos, pr[cols])
        salida.append(pd.DataFrame({
            "fecha_dato": pr["fecha"].values,
            "blue_dato": pr["blue_venta"].values,
            "variacion_predicha_%": yp,
            "blue_predicho": pr["blue_venta"].values * np.exp(yp / 100),
            "blue_real": pr[f"blue_t_mas_{h}"].values,
            "alerta_atipico": atipico(pr),
            "cepo": pr["cepo"].values if "cepo" in pr else np.nan,
            "filas_entrenamiento": len(tr),
        }, index=pr.index))
        if k % 12 == 0:
            print(f"   ... {mes} ({len(tr)} filas de entrenamiento)", flush=True)
    bt = pd.concat(salida)
    # fecha a la que corresponde la predicción = la fecha de la fila t+h (día hábil del dataset)
    fechas = f["fecha"]
    bt["fecha_predicha"] = [fechas.iloc[i + h] if i + h < len(fechas) else fechas.iloc[-1] + pd.offsets.BDay(i + h - len(fechas) + 1)
                            for i in bt.index]
    bt["prediccion_ingenua"] = bt["blue_dato"]
    bt["error_%"] = (bt["blue_predicho"] - bt["blue_real"]) / bt["blue_real"] * 100
    bt["error_ingenuo_%"] = (bt["prediccion_ingenua"] - bt["blue_real"]) / bt["blue_real"] * 100
    return agregar_rangos(bt, h)


def agregar_rangos(bt: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Rango probable conformal y adaptativo: para el día k usa los errores |ln(real/predicho)| de los VENTANA_ERRORES días
    anteriores cuyo resultado ya se conocía (los de k-h o antes). Sin datos futuros.
    """
    bt = bt.copy()
    err = np.abs(np.log(bt["blue_real"] / bt["blue_predicho"])).values
    for cob in COBERTURAS:
        q = np.full(len(bt), np.nan)
        for k in range(len(bt)):
            hasta = k - h + 1                                    # errores conocidos al momento de predecir la fila k
            ventana = err[max(0, hasta - VENTANA_ERRORES):max(hasta, 0)]
            ventana = ventana[~np.isnan(ventana)]
            if len(ventana) >= 40:
                q[k] = np.quantile(ventana, cob)
        pct = int(cob * 100)
        bt[f"rango{pct}_inferior"] = bt["blue_predicho"] * np.exp(-q)
        bt[f"rango{pct}_superior"] = bt["blue_predicho"] * np.exp(q)
        bt[f"real_dentro_rango{pct}"] = np.where(np.isnan(q) | bt["blue_real"].isna(), np.nan,
                                                 ((bt["blue_real"] >= bt[f"rango{pct}_inferior"]) & (bt["blue_real"] <= bt[f"rango{pct}_superior"])).astype(float))
    return bt


# ───────────────────────────── MÉTRICAS ─────────────────────────────
def metricas(bt: pd.DataFrame, h: int) -> pd.DataFrame:
    v = bt.dropna(subset=["blue_real"])
    filas = []

    def fila(nombre, s):
        if len(s) < 5:
            return
        ape_m, ape_n = s["error_%"].abs(), s["error_ingenuo_%"].abs()
        mej, p = c.dm_test(ape_n.reset_index(drop=True), ape_m.reset_index(drop=True), h)
        dir_ok = np.sign(s["variacion_predicha_%"]) == np.sign(np.log(s["blue_real"] / s["blue_dato"]))
        mov = (s["blue_real"] != s["blue_dato"]) & (s["variacion_predicha_%"] != 0)
        filas.append({"periodo": nombre, "dias": len(s), "MAPE_modelo_%": ape_m.mean(), "MAPE_ingenuo_%": ape_n.mean(),
                      "mejora_vs_ingenuo_%": (1 - ape_m.mean() / ape_n.mean()) * 100, "p_DM": p,
                      "MAE_modelo_pesos": (s["blue_predicho"] - s["blue_real"]).abs().mean(),
                      "acierto_direccion_%": dir_ok[mov].mean() * 100 if mov.any() else np.nan,
                      "cobertura_rango80_%": s["real_dentro_rango80"].mean() * 100, "cobertura_rango95_%": s["real_dentro_rango95"].mean() * 100})

    fila("TODO el backtest", v)
    for anio, s in v.groupby(v["fecha_predicha"].dt.year):
        fila(str(anio), s)
    if "cepo" in v:
        fila("con cepo", v[v["cepo"] == 1])
        fila("sin cepo", v[v["cepo"] == 0])
    fila("días con alerta de atípico", v[v["alerta_atipico"] == 1])
    fila("días sin alerta", v[v["alerta_atipico"] == 0])
    return pd.DataFrame(filas)


# ───────────────────────────── PRÓXIMO DÍA ─────────────────────────────
def proximo_dia(f: pd.DataFrame, bt: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    cols = _columnas(f)
    y = f"y_{h}"
    tr = _con_datos(f[(f["fecha"] >= pd.Timestamp(a.FECHA_INICIO_MODELO))].dropna(subset=[y]))
    modelos = entrenar(tr, cols, h)
    ult = f.iloc[[-1]]
    yp = float(predecir(modelos, ult[cols])[0])
    alerta = int(_detector_atipicos(tr, cols)(ult)[0])
    err = np.abs(np.log(bt["blue_real"] / bt["blue_predicho"])).dropna().values[-VENTANA_ERRORES:]
    base = float(ult["blue_venta"].iloc[0])
    pred = base * np.exp(yp / 100)
    out = {"fecha_ultimo_dato": ult["fecha"].iloc[0], "blue_ultimo_dato": base,
           "fecha_predicha": ult["fecha"].iloc[0] + pd.offsets.BDay(h), "blue_predicho": pred, "variacion_predicha_%": yp}
    for cob in COBERTURAS:
        q = np.quantile(err, cob)
        out[f"rango{int(cob*100)}_inferior"], out[f"rango{int(cob*100)}_superior"] = pred * np.exp(-q), pred * np.exp(q)
    out["alerta_alta_incertidumbre"] = alerta
    return pd.DataFrame([out])


# ───────────────────────────── GRÁFICOS ─────────────────────────────
def graficos(bt: pd.DataFrame, h: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("  (matplotlib no disponible:", e, ")")
        return
    DIR_MODELO.mkdir(parents=True, exist_ok=True)
    v = bt.dropna(subset=["blue_real"])
    u = v[v["fecha_predicha"] >= v["fecha_predicha"].max() - pd.DateOffset(months=12)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(u["fecha_predicha"], u["rango95_inferior"], u["rango95_superior"], color="tab:red", alpha=.12, label="Rango 95%")
    ax.fill_between(u["fecha_predicha"], u["rango80_inferior"], u["rango80_superior"], color="tab:red", alpha=.25, label="Rango 80%")
    ax.plot(u["fecha_predicha"], u["blue_real"], color="k", lw=1.4, label="Blue real")
    ax.plot(u["fecha_predicha"], u["blue_predicho"], color="tab:red", lw=1, label=f"Blue predicho ({' + '.join(MODELOS_ENSAMBLE)})")
    al = u[u["alerta_atipico"] == 1]
    ax.scatter(al["fecha_predicha"], al["blue_real"], color="orange", zorder=5, s=25, label="Alerta: día atípico")
    ax.set_ylabel("Pesos por USD"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"Últimos 12 meses: blue predicho con datos hasta {'el día anterior' if h == 1 else f'{h} días hábiles antes'} vs. real")
    fig.tight_layout(); fig.savefig(DIR_MODELO / "09_real_vs_predicho.png", dpi=130); plt.close(fig)

    mens = v.groupby(v["fecha_predicha"].dt.to_period("M")).agg(modelo=("error_%", lambda s: s.abs().mean()),
                                                                 ingenuo=("error_ingenuo_%", lambda s: s.abs().mean()))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = mens.index.to_timestamp()
    ax.plot(x, mens["ingenuo"], color="gray", lw=1.2, ls="--", label="Ingenuo (referencia)")
    ax.plot(x, mens["modelo"], color="tab:red", lw=1.4, label=f"Modelo ML ({' + '.join(MODELOS_ENSAMBLE)})")
    ax.set_ylabel("MAPE del mes (%)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Error mensual de la predicción diaria (menos es mejor)")
    fig.tight_layout(); fig.savefig(DIR_MODELO / "10_error_mensual.png", dpi=130); plt.close(fig)


# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main(ruta=None, h: int = 1, guardar: bool = True):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    ruta = Path(ruta) if ruta else a._buscar()
    d_all = a._leer_csv(ruta)
    d = d_all[d_all["fecha"].dt.dayofweek < 5].reset_index(drop=True)
    f = a.construir_features(d)
    print(f"modelo_diario v{VERSION} | dataset: {ruta} | horizonte h={h} día(s) hábil(es) | modelo: {' + '.join(MODELOS_ENSAMBLE)}")
    print(f"Backtest día por día desde {INICIO_BACKTEST}, reentrenando al inicio de cada mes con todo el pasado disponible...")
    bt = backtest(f, h)
    met = metricas(bt, h)
    sig = proximo_dia(f, bt, h)

    a._titulo(f"RESULTADO DEL BACKTEST ({int(bt['blue_real'].notna().sum())} días predichos)")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(met.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    tot = met.iloc[0]
    print(f"\nLectura: en promedio el valor predicho se equivoca {tot['MAPE_modelo_%']:.2f}% ({tot['MAE_modelo_pesos']:,.1f} pesos por día) "
          f"y el valor real cayó dentro del rango del 80% el {tot['cobertura_rango80_%']:.0f}% de los días "
          f"(dentro del de 95%: {tot['cobertura_rango95_%']:.0f}%).")
    print(f"Frente al ingenuo: mejora {tot['mejora_vs_ingenuo_%']:+.1f}% (p = {tot['p_DM']:.3f}) → {c.veredicto(tot['mejora_vs_ingenuo_%'], tot['p_DM'])}.")

    a._titulo("ÚLTIMOS 10 DÍAS (lo que habría mostrado la app cada día)")
    cols_v = ["fecha_predicha", "blue_predicho", "rango80_inferior", "rango80_superior", "blue_real", "error_%", "alerta_atipico"]
    print(bt.dropna(subset=["blue_real"])[cols_v].tail(10).to_string(index=False, formatters={"fecha_predicha": lambda x: f"{x:%d/%m/%Y}"},
                                                                     float_format=lambda v: f"{v:,.2f}"))
    a._titulo("PREDICCIÓN DEL PRÓXIMO DÍA HÁBIL")
    s = sig.iloc[0]
    print(f"Con datos hasta el {s['fecha_ultimo_dato']:%d/%m/%Y} (blue {s['blue_ultimo_dato']:,.2f}): blue estimado para el {s['fecha_predicha']:%d/%m/%Y} = "
          f"{s['blue_predicho']:,.2f} | rango 80%: {s['rango80_inferior']:,.2f} – {s['rango80_superior']:,.2f} | "
          f"rango 95%: {s['rango95_inferior']:,.2f} – {s['rango95_superior']:,.2f}"
          + (" | ⚠ ALTA INCERTIDUMBRE (día atípico)" if s["alerta_alta_incertidumbre"] else ""))
    print("  (la fecha predicha es el siguiente día hábil de lunes a viernes; no contempla feriados)")

    if guardar:
        orden = ["fecha_dato", "fecha_predicha", "blue_dato", "blue_predicho", "rango80_inferior", "rango80_superior",
                 "rango95_inferior", "rango95_superior", "blue_real", "error_%", "prediccion_ingenua", "error_ingenuo_%",
                 "variacion_predicha_%", "alerta_atipico", "real_dentro_rango80", "real_dentro_rango95", "cepo", "filas_entrenamiento"]
        a._guardar_csv(bt[orden], DIR_MODELO / "predicciones_diarias.csv")
        a._guardar_csv(met, DIR_MODELO / "metricas_backtest.csv")
        a._guardar_csv(sig, DIR_MODELO / "prediccion_proximo_dia.csv")
        graficos(bt, h)
        print(f"\nGuardado en {DIR_MODELO}/ (3 CSV y 2 gráficos). Tiempo: {(time.time()-t0)/60:.1f} min")
    return {"backtest": bt, "metricas": met, "proximo_dia": sig, "features": f}


if __name__ == "__main__":
    main()
