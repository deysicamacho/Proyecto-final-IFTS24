"""Portada: último valor conocido del blue y proyección a dos semanas (10 días hábiles)."""
import base64

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import comun as cm

C = cm.C
AZUL = "#2e8bff"                 # color del modelo en el gráfico (lo anticipado y la proyección): resalta sobre el precio real, en gris
d = cm.cargar()
cm.estilos()
cm.barra_superior(d)

if d["pred"] is None:
    st.error("Todavía no hay proyecciones publicadas. Ejecutá `python actualizar_pipeline.py` (o esperá la actualización "
             "automática) para generar los archivos de data/modelo_h10.")
    st.stop()

p = d["pred"].iloc[0]
alerta = bool(p["alerta_alta_incertidumbre"])
cob = 95 if alerta else 80
ult, f_ult = float(p["blue_ultimo_dato"]), p["fecha_ultimo_dato"]
pred, f_pred = float(p["blue_predicho"]), p["fecha_predicha"]
inf, sup = float(p[f"rango{cob}_inferior"]), float(p[f"rango{cob}_superior"])
serie = cm.serie_blue(d["dataset"])


def delta_html(variacion: float, texto: str = "") -> str:
    if abs(variacion) < 0.05:
        return f"<span class='delta igual'>= {cm.pct(variacion, 1)}</span>{texto}"
    clase, flecha = ("sube", "▲") if variacion > 0 else ("baja", "▼")
    return f"<span class='delta {clase}'>{flecha} {cm.pct(variacion, 1, signo=True)}</span>{texto}"


# ── encabezado ──
st.html("""<div class="titulo">Dólar blue: hoy y <span>en dos semanas</span></div>
<div class="bajada">El último precio de venta conocido y lo que estima un modelo de Machine Learning para dentro de 10 días hábiles.</div>""")

# ── tarjeta 1: último valor conocido ──
previo = serie[serie["fecha"] <= f_ult]
var_2s = (ult / previo["blue_venta"].iloc[-11] - 1) * 100 if len(previo) >= 11 else None
mes = previo.tail(22)["blue_venta"].to_numpy()          # ≈ un mes de días hábiles, para el minigráfico
spark = ""
if len(mes) >= 5:
    a, b = mes.min(), mes.max()
    rng = (b - a) or 1.0
    pts = [(i / (len(mes) - 1) * 300, 58 - (v - a) / rng * 50) for i, v in enumerate(mes)]
    linea = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 64" preserveAspectRatio="none">'
           f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff" stop-opacity=".22"/>'
           f'<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient></defs>'
           f'<polygon points="0,64 {linea} 300,64" fill="url(#g)"/>'
           f'<polyline points="{linea}" fill="none" stroke="#ffffff" stroke-width="2.2" vector-effect="non-scaling-stroke"/></svg>')
    # st.html no admite <svg> directo: se inserta como imagen (data URI)
    spark = (f'<div class="spark"><img alt="Evolución del último mes" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'
             f'<div class="leyenda-rango">Último mes: mínimo {cm.pesos(a)} · máximo {cm.pesos(b)}</div></div>')
linea_2s = (f"<div class='rango'><div class='pie'>{delta_html(var_2s, ' en las últimas dos semanas')}</div>{spark}</div>"
            if var_2s is not None else "")
tarjeta_hoy = f"""<div class="tarjeta">
  <div class="etiqueta">Último valor conocido</div>
  <div class="sub">Dólar blue · precio de venta</div>
  <div class="valor"><span class="moneda">$</span>{cm.num(ult, 0)}</div>
  <div class="pie">Cierre del <b>{cm.fecha_larga(f_ult)}</b></div>
  {linea_2s}
</div>"""

# ── tarjeta 2: proyección a dos semanas, con su rango probable dibujado como una barra ──
var_pred = (pred / ult - 1) * 100
dif = pred - ult
lo = min(inf, ult) - (sup - inf) * 0.12
hi = max(sup, ult) + (sup - inf) * 0.12
pos = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))
veces = "8 de cada 10" if cob == 80 else "19 de cada 20"
tarjeta_proy = f"""<div class="tarjeta proy {'alerta' if alerta else ''}">
  <div class="etiqueta">Proyección a dos semanas</div>
  <div class="sub">Estimación para el <b style="color:var(--texto)">{cm.fecha_larga(f_pred)}</b></div>
  <div class="valor"><span class="moneda">$</span>{cm.num(pred, 0)}</div>
  <div class="pie">{delta_html(var_pred, f" <b>{'+' if dif >= 0 else '−'}{cm.pesos(abs(dif))}</b> respecto del último valor")}</div>
  <div class="rango">
    <div class="etiqueta" style="margin-bottom:6px">Rango probable ({cob}%)</div>
    <div class="cifras"><span>{cm.pesos(inf)}</span><span>{cm.pesos(sup)}</span></div>
    <div class="pista">
      <div class="seg" style="left:{pos(inf):.1f}%; width:{pos(sup) - pos(inf):.1f}%"></div>
      <div class="hoy" style="left:{pos(ult):.1f}%" title="Último valor"></div>
      <div class="punto" style="left:{pos(pred):.1f}%" title="Proyección"></div>
    </div>
    <div class="leyenda-rango">{veces} veces el valor real quedó dentro de este rango. La raya gris marca el último valor.</div>
  </div>
</div>"""
st.html(f"<div class='tarjetas'>{tarjeta_hoy}{tarjeta_proy}</div>")

# ── mensaje del detector de situaciones atípicas, justo después de los valores (sin nombres técnicos) ──
if alerta:
    st.html("""<div class="aviso mal"><div class="icono">⚠️</div><div><b>Alta incertidumbre.</b> Las condiciones del mercado de hoy
      son poco habituales comparadas con la historia con la que aprendió el modelo. Tomá el valor estimado con cautela y usá el
      <b>rango del 95%</b>.</div></div>""")
else:
    st.html("""<div class="aviso ok"><div class="icono">✅</div><div><b>Condiciones habituales.</b> No se detectó nada fuera de lo
      común en los datos de hoy, así que se muestra el rango del 80%.</div></div>""")
for aviso in cm.avisos_datos(d):
    st.html(f"<div class='aviso dato'><div class='icono'>ℹ️</div><div>{aviso}</div></div>")

# ── gráfico de evolución ──
st.html("""<div class="seccion">Evolución del dólar blue</div>
<div class="seccion-sub">Precio real, lo que el modelo había anticipado y la proyección vigente.</div>""")
PERIODOS = {1: "1 mes", 3: "3 meses", 6: "6 meses", 12: "1 año", 0: "Desde 2022"}
meses = st.segmented_control("Período", list(PERIODOS), default=3, format_func=PERIODOS.get, label_visibility="collapsed")
meses = 3 if meses is None else meses
desde = f_ult - pd.DateOffset(months=meses) if meses else pd.Timestamp("2022-01-01")

real = serie[(serie["fecha"] >= desde) & (serie["fecha"] <= f_ult)]
bt = d["bt"]
antic = bt[(bt["fecha_predicha"] >= desde) & (bt["fecha_predicha"] <= f_ult)].sort_values("fecha_predicha") if bt is not None else None
valores = list(real["blue_venta"]) + [inf, sup] + (list(antic["blue_predicho"]) if antic is not None else [])
y_min, y_max = min(valores), max(valores)
margen = (y_max - y_min) * 0.08 or y_max * 0.02
base = y_min - margen
x0 = real["fecha"].min() if len(real) else desde

fig = go.Figure()
fig.add_vrect(x0=f_ult, x1=f_pred + pd.Timedelta(days=1), fillcolor="rgba(46,139,255,0.08)", line_width=0)
fig.add_trace(go.Scatter(x=real["fecha"], y=[base] * len(real), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(
    x=real["fecha"], y=real["blue_venta"], name="Precio real", mode="lines", fill="tonexty",
    fillgradient=dict(type="vertical", colorscale=[[0, "rgba(255,255,255,0.0)"], [1, "rgba(255,255,255,0.06)"]]),
    line=dict(color="rgba(200,212,228,0.6)", width=1.6), hovertemplate="Real: $ %{y:,.0f}<extra></extra>"))
if antic is not None and len(antic):
    fig.add_trace(go.Scatter(
        x=antic["fecha_predicha"], y=antic["blue_predicho"], name="Anticipado por el modelo 2 semanas antes", mode="lines",
        line=dict(color=AZUL, width=3.4), hovertemplate="Anticipado: $ %{y:,.0f}<extra></extra>"))
# proyección vigente: cono del rango, línea, halo y punto en azul intenso
fig.add_trace(go.Scatter(x=[f_ult, f_pred, f_pred, f_ult], y=[ult, sup, inf, ult], mode="lines", fill="toself",
                         fillcolor="rgba(46,139,255,0.25)", line=dict(width=1, color="rgba(46,139,255,0.6)"),
                         name=f"Rango probable ({cob}%)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[f_ult, f_pred], y=[ult, pred], mode="lines", line=dict(color=AZUL, width=4),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[f_ult], y=[ult], mode="markers", marker=dict(size=12, color="#ffffff", line=dict(width=3, color=C["fondo"])),
                         showlegend=False, hovertemplate="Último valor: $ %{y:,.0f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[f_pred], y=[pred], mode="markers", marker=dict(size=40, color="rgba(46,139,255,0.25)"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[f_pred], y=[pred], mode="markers", name="Proyección a dos semanas",
                         marker=dict(size=20, color=AZUL, line=dict(width=4, color="#ffffff")),
                         hovertemplate="Proyección: $ %{y:,.0f}<extra></extra>"))
fig.add_annotation(x=f_pred, y=pred, text=f"<b>Proyección {f_pred:%d/%m}<br>{cm.pesos(pred)}</b>", showarrow=True,
                   arrowcolor=AZUL, arrowwidth=2, ax=-70, ay=-62, bgcolor=AZUL, bordercolor="#ffffff", borderwidth=1.5,
                   borderpad=7, font=dict(color="#ffffff", size=16), align="center")
fig.add_annotation(x=f_ult, y=ult, text=f"{f_ult:%d/%m} · {cm.pesos(ult)}", showarrow=True, arrowcolor=C["suave"],
                   ax=-70, ay=46, bgcolor=C["panel2"], bordercolor=C["borde"], borderpad=4, font=dict(color=C["texto"], size=12))
fig.add_annotation(x=f_ult + (f_pred - f_ult) / 2, y=0.02, yref="paper", yanchor="bottom", text="<b>PROYECCIÓN</b>", showarrow=False,
                   font=dict(color=AZUL, size=11))
fig.update_layout(
    height=470, separators=",.", margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow, sans-serif", color=C["suave"], size=13),
    legend=dict(orientation="h", traceorder="normal", yanchor="bottom", y=1.04, x=0, font=dict(color=C["texto"]), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["borde"], font=dict(color=C["texto"], family="Barlow")),
)
TICKS = {1: ("%d/%m", None), 3: ("%d/%m", None), 6: ("%m/%y", "M1"), 12: ("%m/%y", "M2"), 0: ("%m/%y", "M6")}
fmt, paso = TICKS[meses]
fig.update_xaxes(showgrid=False, tickformat=fmt, dtick=paso, linecolor=C["borde"], range=[x0, f_pred + (f_pred - x0) * 0.05])
fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, tickprefix="$", tickformat=",.0f",
                 range=[base, y_max + margen * 3.5], side="right")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption(f"La línea azul muestra lo que el modelo había estimado dos semanas antes de cada día; la línea gris, el precio "
           f"real que finalmente tuvo el blue. A la derecha, la proyección vigente con su rango probable del {cob}%.")

# ── pie: enlace a la explicación del modelo ──
st.html("<div style='height:14px'></div>")
st.page_link("vistas/modelo.py", label="¿Cómo funciona este pronóstico? Conocé el modelo, qué tan preciso es y qué variables pesan",
             icon=":material/arrow_forward:")
cm.pie_legal(d)
