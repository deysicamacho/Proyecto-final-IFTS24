"""Segunda página: cómo funciona el pronóstico, qué tan preciso fue y qué variables pesan."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import comun as cm

C = cm.C
d = cm.cargar()
cm.estilos()
cm.barra_superior(d)
st.page_link("vistas/inicio.py", label="Volver al pronóstico", icon=":material/arrow_back:")


def estilo(fig: go.Figure, alto: int) -> go.Figure:
    fig.update_layout(height=alto, separators=",.", margin=dict(l=8, r=40, t=10, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow, sans-serif", color=C["suave"], size=13),
                      hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["borde"], font=dict(color=C["texto"])),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=C["texto"])))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, linecolor=C["borde"])
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, linecolor=C["borde"])
    return fig


met, met1 = d["met"], d["met_h1"]
tot = cm.fila_metrica(met, "TODO el backtest")
tot1 = cm.fila_metrica(met1, "TODO el backtest")
m_si = cm.fila_metrica(met, "días con alerta de atípico")
m_no = cm.fila_metrica(met, "días sin alerta")

st.html("""<div class="titulo">Cómo funciona el <span>pronóstico</span></div>
<div class="bajada">Para quien quiera entender qué hay detrás del número: de dónde salen los datos, cómo aprende el modelo,
qué tan preciso fue y qué variables pesan más.</div>""")

# ── 1. En pocas palabras ──
st.html("""<div class="seccion">En pocas palabras</div>
<div class="pasos">
  <div class="paso"><div class="num">1</div><h4>Junta los datos</h4><p>Cada día hábil se descargan el dólar blue, el oficial,
    el MEP y el CCL, y se suman la inflación y los índices de confianza del consumidor y en el Gobierno (Universidad Di Tella).</p></div>
  <div class="paso"><div class="num">2</div><h4>Aprende de la historia</h4><p>Un modelo de Machine Learning estudió más de 1.500
    días desde 2020 para aprender cuánto suele moverse el blue en dos semanas según la brecha, el MEP y el CCL, el cepo,
    las elecciones y el calendario.</p></div>
  <div class="paso"><div class="num">3</div><h4>Estima el movimiento</h4><p>No adivina el precio de cero: estima cuánto va a
    subir o bajar el blue en 10 días hábiles y aplica ese porcentaje al último valor conocido.</p></div>
  <div class="paso"><div class="num">4</div><h4>Mide su propia duda</h4><p>Arma un rango probable con sus errores recientes y
    un segundo algoritmo avisa cuando las condiciones del día son poco habituales.</p></div>
</div>""")

st.html("""<div class="caja"><p><b>¿Qué modelo usa?</b> Una regresión por vectores de soporte (SVR). Se compararon tres
algoritmos (una regresión lineal regularizada, un bosque de árboles de decisión y SVR) y SVR fue el que menos se equivocó con
datos que no había visto. Una forma simple de pensarlo: SVR ignora los movimientos chiquitos del día a día y busca la tendencia
de fondo, lo que lo hace más estable cuando el mercado cambia.</p>
<p class="muted">Se reentrena con los datos nuevos, así que el pronóstico se ajusta a medida que cambia el contexto.</p></div>""")

# ── 2. ¿Qué tan preciso es? ──
st.html("""<div class="seccion">¿Qué tan preciso es?</div>
<div class="seccion-sub">Para medirlo se simuló el uso real: desde enero de 2022, cada mes el modelo se volvió a entrenar solo con
los datos disponibles en ese momento y estimó, día por día, el valor a dos semanas. Después se comparó con lo que realmente pasó.</div>""")
if tot is not None:
    sig = tot["p_DM"] < 0.05 and tot["mejora_vs_ingenuo_%"] > 0
    st.html(f"""<div class="kpis">
      <div class="kpi"><div class="n">{cm.pct(tot['MAPE_modelo_%'], 1)}</div><div class="t">Error promedio a dos semanas
        (unos {cm.pesos(tot['MAE_modelo_pesos'])} por día)</div></div>
      <div class="kpi"><div class="n neutro">{cm.pct(tot['MAPE_ingenuo_%'], 1)}</div><div class="t">Error si se supone que el
        blue no cambia en dos semanas</div></div>
      <div class="kpi"><div class="n">{cm.pct(tot['mejora_vs_ingenuo_%'], 1)}</div><div class="t">Menos error que esa suposición
        {'(la diferencia no se explica por azar)' if sig else '(la diferencia podría deberse al azar)'}</div></div>
      <div class="kpi"><div class="n">{cm.num(tot['cobertura_rango80_%'], 0)}%</div><div class="t">De los días el valor real
        quedó dentro del rango del 80%</div></div>
    </div>""")
    anios = met[met["periodo"].str.fullmatch(r"\d{4}")]
    if len(anios):
        gana = int((anios["MAPE_modelo_%"] < anios["MAPE_ingenuo_%"]).sum())
        fig = go.Figure()
        fig.add_trace(go.Bar(x=anios["periodo"], y=anios["MAPE_ingenuo_%"], name="Si el blue no cambia", marker_color="#4b5f7c",
                             text=anios["MAPE_ingenuo_%"].map(lambda x: cm.pct(x, 1)), textposition="outside",
                             hovertemplate="%{x}: %{y:.2f}%<extra>Si el blue no cambia</extra>"))
        fig.add_trace(go.Bar(x=anios["periodo"], y=anios["MAPE_modelo_%"], name="Modelo", marker_color=C["verde"],
                             text=anios["MAPE_modelo_%"].map(lambda x: cm.pct(x, 1)), textposition="outside",
                             hovertemplate="%{x}: %{y:.2f}%<extra>Modelo</extra>"))
        estilo(fig, 340).update_layout(barmode="group", bargap=0.25, yaxis_title="Error promedio (%)")
        fig.update_traces(textfont=dict(color=C["texto"]), cliponaxis=False)
        st.markdown("**Error promedio por año: cuanto más baja la barra, mejor**")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"El modelo se equivocó menos que la suposición «el blue no cambia» en {gana} de {len(anios)} años. "
                   "Los años con cepo y devaluaciones (2022-2024) fueron más difíciles para cualquier pronóstico.")
    with st.expander("Ver la tabla completa"):
        t = met.copy()
        t["periodo"] = t["periodo"].replace({"TODO el backtest": "Total 2022-2026", "días con alerta de atípico": "Días con alerta",
                                             "días sin alerta": "Días sin alerta"})
        st.dataframe(pd.DataFrame({
            "Período": t["periodo"], "Días": t["dias"].astype(int),
            "Error del modelo": t["MAPE_modelo_%"].map(lambda x: cm.pct(x, 2)),
            "Error si el blue no cambia": t["MAPE_ingenuo_%"].map(lambda x: cm.pct(x, 2)),
            "Mejora": t["mejora_vs_ingenuo_%"].map(lambda x: cm.pct(x, 1, signo=True)),
            "¿Significativa?": np.where((t["p_DM"] < 0.05) & (t["mejora_vs_ingenuo_%"] > 0), "Sí", "No"),
            "Dentro del rango 80%": t["cobertura_rango80_%"].map(lambda x: cm.pct(x, 0)),
            "Dentro del rango 95%": t["cobertura_rango95_%"].map(lambda x: cm.pct(x, 0)),
        }), hide_index=True, width="stretch")
        st.caption("Error = error porcentual absoluto medio (MAPE). «Significativa» usa la prueba de Diebold-Mariano (p < 0,05): "
                   "indica si la mejora frente a suponer que el blue no cambia puede deberse al azar.")
if tot1 is not None:
    st.html(f"""<div class="caja"><p><b>¿Y para mañana?</b> Para el día siguiente el modelo no mejora a suponer que el blue no
      cambia (error de {cm.pct(tot1['MAPE_modelo_%'], 2)} contra {cm.pct(tot1['MAPE_ingenuo_%'], 2)}): de un día para otro el
      blue casi no se mueve. Por eso la portada muestra solo la proyección a dos semanas, donde el modelo sí aporta.</p></div>""")

# ── 3. ¿Qué variables pesan? ──
st.html("""<div class="seccion">¿Qué variables pesan más?</div>
<div class="seccion-sub">Se midió cuánto empeora el pronóstico si se «desordena» al azar la información de cada grupo o variable.
Cuanto más larga la barra, más depende el modelo de esa información.</div>""")
imp = d["imp"]
if imp is None:
    st.info("La importancia de las variables se publica cuando se ejecuta la comparación de algoritmos (los lunes).")
else:
    g = imp[(imp["horizonte"] == cm.H) & imp["variable"].str.startswith("[GRUPO]")].copy()
    g["nombre"] = g["grupo"].map(cm.GRUPOS).fillna(g["grupo"])
    g = g.sort_values("aumento_MAPE_puntos")
    top = imp[(imp["horizonte"] == cm.H) & ~imp["variable"].str.startswith("[GRUPO]")].nlargest(8, "aumento_MAPE_puntos")
    top = top.assign(nombre=top["variable"].map(cm.NOMBRES).fillna(top["variable"])).iloc[::-1]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Por grupo**")
        fig = go.Figure(go.Bar(x=g["aumento_MAPE_puntos"], y=g["nombre"], orientation="h", marker_color=C["verde"],
                               text=g["aumento_MAPE_puntos"].map(lambda x: cm.num(x, 2)), textposition="outside",
                               textfont=dict(color=C["texto"]), cliponaxis=False,
                               hovertemplate="%{y}: +%{x:.3f} puntos de error<extra></extra>"))
        st.plotly_chart(estilo(fig, 300).update_layout(xaxis_title="Aumento del error (puntos)"), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        st.markdown("**Las 8 variables con más peso**")
        fig = go.Figure(go.Bar(x=top["aumento_MAPE_puntos"], y=top["nombre"], orientation="h", marker_color=C["celeste"],
                               text=top["aumento_MAPE_puntos"].map(lambda x: cm.num(x, 3)), textposition="outside",
                               textfont=dict(color=C["texto"]), cliponaxis=False,
                               hovertemplate="%{y}: +%{x:.3f} puntos de error<extra></extra>"))
        st.plotly_chart(estilo(fig, 300).update_layout(xaxis_title="Aumento del error (puntos)"), width="stretch",
                        config={"displayModeBar": False})
    menor = g.iloc[0]["grupo"] if len(g) else ""
    EN_FRASE = {"D_regimen_y_calendario": "el cepo, las elecciones y el calendario", "E_mep_ccl": "el dólar MEP y el CCL",
                "A_precio_y_brecha": "el precio del blue y su brecha con el oficial",
                "C_mensuales_sentimiento_inflacion": "la confianza y la inflación",
                "B_mercado_diario": "las reservas, el riesgo país, las tasas y la bolsa"}
    orden = [EN_FRASE.get(x, x) for x in g["grupo"].iloc[::-1]]
    lineas = [f"<p>Lo que más pesa es <b>{orden[0]}</b>"
              + (", seguido por " + " y ".join(f"<b>{x}</b>" for x in orden[1:3]) if len(orden) >= 3 else "") + ".</p>"] if orden else []
    if len(top) and top.iloc[-1]["variable"] == "mes_aguinaldo":
        lineas.append("<p>La variable individual más importante es el <b>mes de aguinaldo</b> (junio, julio y diciembre): en esos "
                      "meses el blue tuvo un comportamiento distinto al del resto del año. Ojo: el modelo detecta el patrón, no su causa.</p>")
    if menor == "C_mensuales_sentimiento_inflacion":
        lineas.append("<p>Los índices de <b>confianza</b> (del consumidor y en el Gobierno) y la <b>inflación</b> forman parte del "
                      "modelo, pero son el grupo que menos pesa a dos semanas.</p>")
    lineas.append("<p class='muted'>Las reservas del BCRA, el riesgo país, las tasas y la bolsa se probaron y empeoraban el "
                  "pronóstico, por eso no forman parte del modelo final.</p>")
    st.html(f"<div class='caja'>{''.join(lineas)}</div>")

# ── 4. Rango probable y alerta ──
st.html("<div class='seccion'>El rango probable y la alerta</div>")
cob80 = f" (en la simulación, el {cm.num(tot['cobertura_rango80_%'], 0)}% de los días)" if tot is not None else ""
cob95 = f" ({cm.num(tot['cobertura_rango95_%'], 0)}%)" if tot is not None else ""
alerta_txt = (f" En la simulación 2022-2026, en los días con alerta el error promedio a dos semanas fue de "
              f"{cm.pct(m_si['MAPE_modelo_%'], 1)}, contra {cm.pct(m_no['MAPE_modelo_%'], 1)} en los días normales."
              if m_si is not None and m_no is not None else "")
st.html(f"""<div class="pasos">
  <div class="paso"><h4>📏 Rango probable</h4><p>Se arma con los errores que el modelo cometió en los últimos 120 días: si se
    venía equivocando poco, el rango se achica; si se venía equivocando mucho, se agranda. El rango del 80% contiene el valor
    real 8 de cada 10 veces{cob80}; el del 95%, 19 de cada 20{cob95}.</p></div>
  <div class="paso"><h4>🚨 Alerta de alta incertidumbre</h4><p>Un segundo algoritmo, llamado Isolation Forest, compara las
    condiciones de hoy con las de toda la historia. Si son poco habituales (como en una devaluación o una elección), la portada
    muestra la alerta y el rango del 95%.{alerta_txt}</p></div>
</div>""")

# ── 5. Datos y límites ──
st.html("""<div class="seccion">De dónde salen los datos</div>
<div class="caja"><p><b>Cotizaciones del dólar</b> (blue, oficial, MEP y CCL): MonedAPI. <b>Inflación</b>: ArgentinaDatos.
<b>Tasas</b>: API del Banco Central. <b>Índices de confianza</b> del consumidor y en el Gobierno: Universidad Torcuato Di Tella.</p>
<p class="muted">Cada dato se usa recién desde el día en que se hizo público, para que la simulación no «haga trampa» con información
que en ese momento no se conocía.</p></div>
<div class="seccion">Límites</div>
<div class="caja"><p>El modelo aprende de patrones del pasado: no puede anticipar hechos imprevistos como un anuncio, una renuncia o
una crisis externa. En los meses de devaluación brusca el error fue mucho mayor que el promedio.</p>
<p>A un mes (20 días hábiles) la mejora no fue confiable, por eso no se publica.</p>
<p class="muted">Es una herramienta informativa de un proyecto académico. No es una recomendación de compra o venta.</p></div>""")

st.html("<div style='height:14px'></div>")
st.page_link("vistas/inicio.py", label="Volver al pronóstico del dólar blue", icon=":material/arrow_back:")
cm.pie_legal(d)
est = d["estado"]
if est.get("versiones"):
    st.caption("Versiones: " + ", ".join(f"{k} {v}" for k, v in est["versiones"].items()) + f" · app {cm.VERSION_APP}")
