"""
Aplicación web del proyecto: predicción del dólar blue (Streamlit).  -- versión 1.0

Muestra lo que produce el pipeline (actualizar_pipeline.py → modelo_diario.py):
  - el valor estimado del blue para el próximo día hábil y a dos semanas, con su rango probable del 80% y 95%;
  - la ALERTA DE ALTA INCERTIDUMBRE cuando Isolation Forest marca el día como atípico (y entonces destaca el rango del 95%);
  - la evolución real frente a lo predicho, la precisión histórica contra el modelo ingenuo y el peso de cada grupo de variables.
No entrena ni descarga nada: lee los CSV de data/, que se actualizan solos cada día hábil (GitHub Actions).

Ejecutar en una computadora:   streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

VERSION = "1.0"
DATA = Path(__file__).parent / "data"
ZONA = ZoneInfo("America/Argentina/Buenos_Aires")
LEER = dict(sep=";", decimal=",", encoding="utf-8-sig")
HORIZONTES = {1: "Próximo día hábil", 10: "En dos semanas (10 días hábiles)"}
DIAS_DATO_VIEJO = 5          # días corridos sin dato nuevo para avisar que la información está desactualizada

GRUPOS = {"A_precio_y_brecha": "Precio del blue y brecha con el oficial",
          "C_mensuales_sentimiento_inflacion": "Confianza (ICC, ICG) e inflación",
          "D_regimen_y_calendario": "Cepo, elecciones y calendario",
          "E_mep_ccl": "Dólar MEP y CCL",
          "B_mercado_diario": "Reservas, riesgo país, tasas y bolsa"}
NOMBRES = {
    "mes_aguinaldo": "Mes de aguinaldo (junio, julio o diciembre)", "dia_mes": "Día del mes", "dia_semana": "Día de la semana",
    "cepo": "Vigencia del cepo", "dias_a_eleccion": "Días hasta la próxima elección", "dias_desde_eleccion": "Días desde la última elección",
    "brecha_pct": "Brecha blue-oficial", "brecha_z_120": "Brecha comparada con los últimos 6 meses", "brecha_cambio_5": "Cambio de la brecha en 5 días",
    "brecha_cambio_20": "Cambio de la brecha en 20 días", "regimen_brecha_alta": "Brecha mayor al 20%",
    "brecha_blue_mep_pct": "Diferencia blue-MEP", "brecha_blue_ccl_pct": "Diferencia blue-CCL",
    "brecha_blue_mep_cambio_5": "Cambio de la diferencia blue-MEP en 5 días", "brecha_blue_ccl_cambio_5": "Cambio de la diferencia blue-CCL en 5 días",
    "canje_ccl_mep_pct": "Diferencia CCL-MEP", "icg_t1": "Confianza en el Gobierno (mes anterior)", "icc_t1": "Confianza del consumidor (mes anterior)",
    "d_icg_t1": "Cambio de la confianza en el Gobierno", "d_icc_t1": "Cambio de la confianza del consumidor",
    "infl_t1": "Inflación del mes anterior", "infl_acel_t1": "Aceleración de la inflación", "tasa_real_mensual": "Tasa real del plazo fijo",
    "vol_blue_20": "Volatilidad del blue (1 mes)", "vol_blue_60": "Volatilidad del blue (3 meses)",
    "dist_media_blue_20": "Distancia del blue a su promedio de 1 mes", "dist_media_blue_60": "Distancia del blue a su promedio de 3 meses",
}
for k in (1, 5, 10, 20, 60):
    NOMBRES[f"ret_blue_{k}"] = f"Variación del blue en {k} día{'s' if k > 1 else ''}"
for k in (5, 20):
    NOMBRES[f"ret_oficial_{k}"] = f"Variación del oficial en {k} días"
for k in (1, 5, 20):
    NOMBRES[f"ret_mep_{k}"] = f"Variación del MEP en {k} día{'s' if k > 1 else ''}"
    NOMBRES[f"ret_ccl_{k}"] = f"Variación del CCL en {k} día{'s' if k > 1 else ''}"

COLOR = {"real": "#1f2937", "pred": "#2563eb", "banda": "rgba(37,99,235,0.15)", "alerta": "#dc2626", "ingenuo": "#9ca3af"}


# ───────────────────────────── FORMATO ─────────────────────────────
def num(x: float, dec: int = 2) -> str:
    """Formato argentino: 1.573,10"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pesos(x: float, dec: int = 2) -> str:
    """Para textos con Markdown: el signo $ va escapado (dos $ seguidos Streamlit los toma como fórmula)."""
    return f"\\$ {num(x, dec)}"


def pesos_valor(x: float, dec: int = 2) -> str:
    """Para st.metric (sin Markdown)."""
    return f"$ {num(x, dec)}"


def pct(x: float, dec: int = 1, signo: bool = False) -> str:
    return ("+" if signo and x > 0 else "") + num(x, dec) + "%"


# ───────────────────────────── DATOS ─────────────────────────────
def _mtime(ruta: Path) -> float:
    return ruta.stat().st_mtime if ruta.exists() else 0.0


@st.cache_data(show_spinner=False)
def _csv(ruta: str, fechas: tuple, _mtime: float) -> pd.DataFrame | None:
    p = Path(ruta)
    if not p.exists():
        return None
    df = pd.read_csv(p, **LEER)
    for c in fechas:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
    return df


def csv(ruta: Path, fechas: tuple = ()) -> pd.DataFrame | None:
    return _csv(str(ruta), fechas, _mtime(ruta))


def cargar() -> dict:
    d = {"pred": {}, "bt": {}, "met": {}}
    for h in HORIZONTES:
        carpeta = DATA / f"modelo_h{h}"
        d["pred"][h] = csv(carpeta / "prediccion_proximo_dia.csv", ("fecha_ultimo_dato", "fecha_predicha"))
        d["bt"][h] = csv(carpeta / "predicciones_diarias.csv", ("fecha_dato", "fecha_predicha"))
        d["met"][h] = csv(carpeta / "metricas_backtest.csv")
    d["imp"] = csv(DATA / "modelos" / "importancia_variables.csv")
    d["dataset"] = csv(DATA / "dataset_dolar_blue.csv", ("fecha",))
    est = DATA / "app" / "estado.json"
    d["estado"] = json.loads(est.read_text(encoding="utf-8")) if est.exists() else {}
    return d


def fila_metrica(met: pd.DataFrame | None, periodo: str) -> pd.Series | None:
    if met is None:
        return None
    r = met[met["periodo"] == periodo]
    return r.iloc[0] if len(r) else None


def calidad_mep_ccl(ds: pd.DataFrame | None) -> list[str]:
    """Aviso si el MEP o el CCL repiten el mismo valor varios días hábiles seguidos (hueco de la fuente)."""
    avisos = []
    if ds is None:
        return avisos
    hab = ds[ds["fecha"].dt.dayofweek < 5]
    for col, nombre in (("dolar_mep", "MEP"), ("dolar_ccl", "CCL")):
        s = hab[col].dropna().tail(6) if col in hab else pd.Series(dtype=float)
        if len(s) == 6 and s.nunique() == 1:
            avisos.append(f"El dólar {nombre} repite el mismo valor ({pesos(s.iloc[-1])}) en los últimos 6 días hábiles: "
                          "puede ser un hueco de la fuente de datos y la predicción podría ser menos precisa.")
    return avisos


# ───────────────────────────── BLOQUES DE LA PÁGINA ─────────────────────────────
def bloque_alerta(d: dict) -> bool:
    """Alerta de alta incertidumbre basada en Isolation Forest (columna alerta_alta_incertidumbre de modelo_diario.py)."""
    alertas = {h: int(p["alerta_alta_incertidumbre"].iloc[0]) for h, p in d["pred"].items() if p is not None}
    hay = any(alertas.values())
    m_si = fila_metrica(d["met"].get(10), "días con alerta de atípico")
    m_no = fila_metrica(d["met"].get(10), "días sin alerta")
    contexto = ""
    if m_si is not None and m_no is not None:
        contexto = (f" En la simulación 2022-2026, en los días con alerta el error promedio a dos semanas fue de "
                    f"{pct(m_si['MAPE_modelo_%'], 1)}, contra {pct(m_no['MAPE_modelo_%'], 1)} en los días normales.")
    if hay:
        cuales = " y ".join(HORIZONTES[h].split(" (")[0].lower() for h, v in alertas.items() if v)
        st.error(f"**⚠️ ALTA INCERTIDUMBRE** ({cuales}). El detector de anomalías (Isolation Forest) encontró que las condiciones "
                 f"del mercado de hoy son poco habituales comparadas con la historia con la que se entrenó el modelo. "
                 f"Tomá el valor estimado con cautela y usá el **rango del 95%**.{contexto}")
    else:
        st.success(f"**✅ Condiciones habituales.** El detector de anomalías (Isolation Forest) no encontró nada atípico en los "
                   f"datos de hoy, así que se muestra el rango del 80%.{contexto}")
    return hay


def tarjeta(col, h: int, p: pd.DataFrame, met: pd.DataFrame | None):
    s = p.iloc[0]
    alerta = bool(s["alerta_alta_incertidumbre"])
    cob = 95 if alerta else 80
    with col:
        st.metric(f"{HORIZONTES[h]} · {s['fecha_predicha']:%d/%m/%Y}", pesos_valor(s["blue_predicho"]),
                  delta=f"{pct(s['variacion_predicha_%'], 2, signo=True)} vs. último valor", delta_color="off")
        st.markdown(f"Rango probable del **{cob}%**: {pesos(s[f'rango{cob}_inferior'])} a {pesos(s[f'rango{cob}_superior'])}"
                    + ("  \n⚠️ *alta incertidumbre*" if alerta else ""))
        otro = 80 if alerta else 95
        st.caption(f"Rango del {otro}%: {pesos(s[f'rango{otro}_inferior'])} a {pesos(s[f'rango{otro}_superior'])}")
        tot = fila_metrica(met, "TODO el backtest")
        if tot is not None:
            sig = tot["p_DM"] < 0.05 and tot["mejora_vs_ingenuo_%"] > 0
            if sig:
                st.caption(f"✔ En 2022-2026 este pronóstico redujo el error un {pct(tot['mejora_vs_ingenuo_%'], 1)} frente a "
                           f"suponer que el blue no cambia (diferencia significativa, p = {num(tot['p_DM'], 3)}).")
            else:
                st.caption(f"ℹ️ En 2022-2026 este pronóstico **no** mejoró de forma significativa a suponer que el blue no cambia "
                           f"(error {pct(tot['MAPE_modelo_%'], 2)} contra {pct(tot['MAPE_ingenuo_%'], 2)}). Usalo como referencia.")


def grafico_evolucion(bt: pd.DataFrame, p: pd.DataFrame | None, h: int, meses: int, cob: int, marcar: bool) -> go.Figure:
    desde = bt["fecha_predicha"].max() - pd.DateOffset(months=meses) if meses else bt["fecha_predicha"].min()
    v = bt[bt["fecha_predicha"] >= desde].sort_values("fecha_predicha")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v["fecha_predicha"], y=v[f"rango{cob}_superior"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=v["fecha_predicha"], y=v[f"rango{cob}_inferior"], line=dict(width=0), fill="tonexty",
                             fillcolor=COLOR["banda"], name=f"Rango del {cob}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=v["fecha_predicha"], y=v["blue_real"], name="Valor real", line=dict(color=COLOR["real"], width=2),
                             hovertemplate="%{x|%d/%m/%Y}<br>Real: $ %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=v["fecha_predicha"], y=v["blue_predicho"], name=f"Predicho {h} día{'s' if h > 1 else ''} antes",
                             line=dict(color=COLOR["pred"], width=1.6),
                             hovertemplate="%{x|%d/%m/%Y}<br>Predicho: $ %{y:,.2f}<extra></extra>"))
    if marcar:
        al = v[v["alerta_atipico"] == 1]
        if len(al):
            fig.add_trace(go.Scatter(x=al["fecha_predicha"], y=al["blue_predicho"], mode="markers", name="Día con alerta",
                                     marker=dict(color=COLOR["alerta"], size=8, symbol="x"),
                                     hovertemplate="%{x|%d/%m/%Y}<br>Alerta de alta incertidumbre<extra></extra>"))
    if p is not None:
        s = p.iloc[0]
        fig.add_trace(go.Scatter(x=[s["fecha_predicha"]], y=[s["blue_predicho"]], mode="markers", name="Predicción vigente",
                                 marker=dict(color=COLOR["alerta"] if s["alerta_alta_incertidumbre"] else COLOR["pred"], size=12,
                                             symbol="diamond"),
                                 error_y=dict(type="data", symmetric=False, array=[s[f"rango{cob}_superior"] - s["blue_predicho"]],
                                              arrayminus=[s["blue_predicho"] - s[f"rango{cob}_inferior"]]),
                                 hovertemplate="%{x|%d/%m/%Y}<br>Predicción: $ %{y:,.2f}<extra></extra>"))
    fig.update_layout(separators=",.", height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Pesos por dólar", plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", tickformat="%d/%m/%y")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def tabla_precision(met: pd.DataFrame) -> pd.DataFrame:
    t = met[~met["periodo"].isin(["con cepo", "sin cepo"])].copy()
    t["¿Diferencia significativa?"] = np.where((t["p_DM"] < 0.05) & (t["mejora_vs_ingenuo_%"] > 0), "Sí, a favor del modelo",
                                              np.where((t["p_DM"] < 0.05), "Sí, en contra", "No"))
    out = pd.DataFrame({
        "Período": t["periodo"].replace({"TODO el backtest": "Total 2022-2026", "días con alerta de atípico": "Días con alerta",
                                          "días sin alerta": "Días sin alerta"}),
        "Días": t["dias"].astype(int),
        "Error del modelo": t["MAPE_modelo_%"].map(lambda x: pct(x, 2)),
        "Error si el blue no cambia": t["MAPE_ingenuo_%"].map(lambda x: pct(x, 2)),
        "Mejora": t["mejora_vs_ingenuo_%"].map(lambda x: pct(x, 1, signo=True)),
        "¿Diferencia significativa?": t["¿Diferencia significativa?"],
        "Real dentro del rango 80%": t["cobertura_rango80_%"].map(lambda x: pct(x, 0)),
        "Real dentro del rango 95%": t["cobertura_rango95_%"].map(lambda x: pct(x, 0)),
    })
    return out


def grafico_grupos(imp: pd.DataFrame, h: int) -> go.Figure | None:
    g = imp[(imp["horizonte"] == h) & imp["variable"].str.startswith("[GRUPO]")].copy()
    if g.empty:
        return None
    g["nombre"] = g["grupo"].map(GRUPOS).fillna(g["grupo"])
    g = g.sort_values("aumento_MAPE_puntos")
    fig = go.Figure(go.Bar(x=g["aumento_MAPE_puntos"], y=g["nombre"], orientation="h", marker_color=COLOR["pred"],
                           text=g["aumento_MAPE_puntos"].map(lambda x: num(x, 3)), textposition="outside",
                           hovertemplate="%{y}: +%{x:.3f} puntos de error<extra></extra>"))
    fig.update_layout(separators=",.", height=280, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="white",
                      xaxis_title="Cuánto aumenta el error si se mezcla al azar la información del grupo (puntos de MAPE)")
    return fig


# ───────────────────────────── PÁGINA ─────────────────────────────
def main():
    st.set_page_config(page_title="Dólar blue: valor estimado", page_icon="💵", layout="wide")
    d = cargar()
    if any(d["pred"][h] is None for h in HORIZONTES):
        st.title("💵 Dólar blue: valor estimado")
        st.error("Todavía no hay predicciones publicadas. Ejecutá `python actualizar_pipeline.py` (o esperá la actualización "
                 "automática) para generar los archivos de data/modelo_h1 y data/modelo_h10.")
        st.stop()

    p1 = d["pred"][1].iloc[0]
    ahora = datetime.now(ZONA)
    st.title("💵 Dólar blue: valor estimado")
    act = d["estado"].get("actualizado")
    act_txt = f" · Actualizado el {datetime.fromisoformat(act):%d/%m/%Y a las %H:%M}" if act else ""
    st.caption(f"Último dato del blue: **{p1['fecha_ultimo_dato']:%d/%m/%Y}** ({pesos(p1['blue_ultimo_dato'])}){act_txt}. "
               "Modelo de Machine Learning (SVR) con variables cambiarias, de sentimiento y de contexto. "
               "Herramienta informativa: no es asesoramiento financiero.")

    # ── avisos de calidad del dato ──
    dias = (pd.Timestamp(ahora.date()) - p1["fecha_ultimo_dato"]).days
    if dias > DIAS_DATO_VIEJO:
        st.warning(f"Los datos no se actualizan desde hace {dias} días: el valor estimado puede estar desactualizado.")
    for aviso in calidad_mep_ccl(d["dataset"]) + d["estado"].get("avisos", []):
        st.warning(aviso)

    # ── alerta de Isolation Forest y valores estimados ──
    bloque_alerta(d)
    c0, c1, c2 = st.columns(3)
    with c0:
        st.metric(f"Último valor conocido · {p1['fecha_ultimo_dato']:%d/%m/%Y}", pesos_valor(p1["blue_ultimo_dato"]))
        st.caption("Precio de venta del dólar blue. Las predicciones parten de este valor y estiman cuánto se moverá.")
    for col, h in ((c1, 1), (c2, 10)):
        tarjeta(col, h, d["pred"][h], d["met"][h])

    st.divider()
    t1, t2, t3, t4 = st.tabs(["📈 Evolución", "🎯 ¿Qué tan preciso es?", "🧩 ¿Qué variables pesan?", "ℹ️ Cómo funciona"])

    with t1:
        a, b, c_, e = st.columns([1.3, 1.2, 1, 1])
        h = a.radio("Horizonte", list(HORIZONTES), index=1, format_func=lambda x: "1 día" if x == 1 else "10 días", horizontal=True)
        meses = b.selectbox("Período", [3, 6, 12, 24, 0], index=1, format_func=lambda m: "Todo (desde 2022)" if m == 0 else f"Últimos {m} meses")
        alerta_h = bool(d["pred"][h]["alerta_alta_incertidumbre"].iloc[0])
        cob = c_.radio("Rango", [80, 95], index=1 if alerta_h else 0, format_func=lambda x: f"{x}%", horizontal=True)
        marcar = e.toggle("Marcar días con alerta", value=True)
        bt = d["bt"][h]
        if bt is not None:
            st.plotly_chart(grafico_evolucion(bt, d["pred"][h], h, meses, cob, marcar), width="stretch")
            st.caption(f"Cada punto azul es lo que la herramienta habría predicho {h} día{'s' if h > 1 else ''} hábil{'es' if h > 1 else ''} "
                       "antes, entrenada solo con datos de ese momento (simulación día por día desde 2022). La banda es el rango probable; "
                       "las cruces rojas son días en que Isolation Forest emitió la alerta de alta incertidumbre.")

    with t2:
        h2_ = st.radio("Horizonte ", list(HORIZONTES), index=1, format_func=lambda x: "1 día" if x == 1 else "10 días",
                       horizontal=True, key="h_prec")
        met = d["met"][h2_]
        if met is not None:
            tot = fila_metrica(met, "TODO el backtest")
            st.markdown(f"En la simulación 2022-2026 ({int(tot['dias'])} días), el valor estimado se equivocó en promedio un "
                        f"**{pct(tot['MAPE_modelo_%'], 2)}** ({pesos(tot['MAE_modelo_pesos'])} por día). Suponer que el blue no cambia "
                        f"se equivocó un **{pct(tot['MAPE_ingenuo_%'], 2)}**. El valor real cayó dentro del rango del 80% el "
                        f"{pct(tot['cobertura_rango80_%'], 0)} de los días y dentro del de 95% el {pct(tot['cobertura_rango95_%'], 0)}.")
            st.dataframe(tabla_precision(met), hide_index=True, width="stretch")
            st.caption("Error = error porcentual absoluto medio (MAPE). «Diferencia significativa» usa la prueba de Diebold-Mariano "
                       "(p < 0,05): indica si la mejora frente a suponer que el blue no cambia puede deberse al azar.")

    with t3:
        imp = d["imp"]
        if imp is None:
            st.info("La importancia de las variables se publica cuando se ejecuta la comparación de algoritmos (los lunes).")
        else:
            st.markdown("**Peso de cada grupo de variables en la predicción a dos semanas.** Se mide cuánto aumenta el error si se "
                        "mezcla al azar la información del grupo (importancia por permutación).")
            fig = grafico_grupos(imp, 10)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            top = imp[(imp["horizonte"] == 10) & ~imp["variable"].str.startswith("[GRUPO]")].nlargest(8, "aumento_MAPE_puntos")
            st.markdown("**Las 8 variables con más peso**")
            st.dataframe(pd.DataFrame({"Variable": top["variable"].map(NOMBRES).fillna(top["variable"]),
                                       "Grupo": top["grupo"].map(GRUPOS).fillna(top["grupo"]),
                                       "Aumento del error (puntos)": top["aumento_MAPE_puntos"].map(lambda x: num(x, 3))}),
                         hide_index=True, width="stretch")
            st.caption("Las reservas, el riesgo país, las tasas y la bolsa se evaluaron y no mejoraron la predicción, por eso no "
                       "forman parte del modelo final.")

    with t4:
        st.markdown(
            "**Qué hace la herramienta.** Cada día hábil estima el precio de venta del dólar blue para el día hábil siguiente y para "
            "dentro de dos semanas, con un rango probable y una alerta cuando la situación es poco habitual.\n\n"
            "**De dónde salen los datos.** MonedAPI (blue, oficial, MEP y CCL), ArgentinaDatos (inflación y riesgo país), la API del "
            "BCRA (tasas y reservas) y los índices de Confianza del Consumidor y en el Gobierno de la Universidad Torcuato Di Tella. "
            "Cada dato se usa recién desde el día en que fue público.\n\n"
            "**Cómo se calcula.** Un modelo de Machine Learning (SVR, regresión por vectores de soporte) aprende de la historia desde "
            "2020 cuánto suele moverse el blue según la brecha con el oficial, el MEP y el CCL, su dinámica reciente, la confianza, "
            "la inflación, el cepo, las elecciones y el calendario (por ejemplo, el aguinaldo). Se reentrena con los datos nuevos.\n\n"
            "**Rango probable.** Se arma con los errores que el modelo cometió en los últimos 120 días: el rango del 80% contiene el "
            "valor real 8 de cada 10 veces; el del 95%, 19 de cada 20.\n\n"
            "**Alerta de alta incertidumbre.** Un segundo algoritmo, Isolation Forest, compara las condiciones de hoy con las de la "
            "historia de entrenamiento. Si son poco habituales (como en una devaluación o una elección), muestra la alerta y "
            "recomienda el rango del 95%, porque en esos días el error suele ser mucho mayor.\n\n"
            "**Límites.** El modelo aprende de patrones del pasado y no puede anticipar hechos imprevistos (un anuncio, una renuncia, "
            "una crisis externa). Para el día siguiente no mejora a suponer que el blue no cambia; su ventaja aparece a dos semanas.\n\n"
            "**Aviso.** Es una herramienta informativa de un proyecto académico. No es una recomendación de compra o venta.")
        est = d["estado"]
        if est:
            st.caption("Versiones: " + ", ".join(f"{k} {v}" for k, v in est.get("versiones", {}).items()) + f" · app {VERSION}")


main()
