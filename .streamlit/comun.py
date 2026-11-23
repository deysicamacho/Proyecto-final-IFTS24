"""
Funciones compartidas por las dos páginas de la aplicación (versión 2.1): lectura de datos, formato argentino y estilos.
No entrena ni descarga nada: lee los CSV de data/ que deja actualizar_pipeline.py.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

VERSION_APP = "2.1"
DATA = Path(__file__).parent / "data"
ZONA = ZoneInfo("America/Argentina/Buenos_Aires")
LEER = dict(sep=";", decimal=",", encoding="utf-8-sig")
H = 10                       # horizonte publicado: dos semanas (10 días hábiles)
DIAS_DATO_VIEJO = 5          # días corridos sin dato nuevo para avisar que la información está desactualizada
DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

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
for _k in (1, 5, 10, 20, 60):
    NOMBRES[f"ret_blue_{_k}"] = f"Variación del blue en {_k} día{'s' if _k > 1 else ''}"
for _k in (5, 20):
    NOMBRES[f"ret_oficial_{_k}"] = f"Variación del oficial en {_k} días"
for _k in (1, 5, 20):
    NOMBRES[f"ret_mep_{_k}"] = f"Variación del MEP en {_k} día{'s' if _k > 1 else ''}"
    NOMBRES[f"ret_ccl_{_k}"] = f"Variación del CCL en {_k} día{'s' if _k > 1 else ''}"

# Paleta (fondo azul noche y acento verde, en la línea de los portales de cotizaciones)
C = {"fondo": "#0b1626", "panel": "#12233a", "panel2": "#172c48", "borde": "#24406a", "texto": "#e9eff7", "suave": "#95a7c0",
     "verde": "#19c39c", "verde_osc": "#079a7d", "rojo": "#ff5d6c", "ambar": "#f4b740", "celeste": "#5fb6ff"}


# ───────────────────────────── FORMATO ─────────────────────────────
def num(x, dec: int = 2) -> str:
    """Formato argentino: 1.573,10"""
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pesos(x, dec: int = 0) -> str:
    """Para HTML (st.html): el signo $ no necesita escaparse."""
    return f"${num(x, dec)}"


def pesos_md(x, dec: int = 0) -> str:
    """Para st.markdown: el $ se escribe como entidad para que Streamlit no lo tome como fórmula."""
    return f"&#36;{num(x, dec)}"


def pct(x, dec: int = 1, signo: bool = False) -> str:
    return ("+" if signo and x > 0 else "") + num(x, dec) + "%"


def fecha_larga(f: pd.Timestamp) -> str:
    return f"{DIAS_SEMANA[f.dayofweek]} {f:%d/%m/%Y}"


# ───────────────────────────── DATOS ─────────────────────────────
def _firma(ruta: Path) -> tuple:
    """Fecha de modificación y tamaño del archivo: si cambian, la caché se renueva y se leen los datos nuevos."""
    if not ruta.exists():
        return (0.0, 0)
    s = ruta.stat()
    return (s.st_mtime, s.st_size)


# En st.cache_data los parámetros que empiezan con "_" NO forman parte de la clave: por eso la firma se llama "firma".
@st.cache_data(show_spinner=False)
def _csv(ruta: str, fechas: tuple, firma: tuple) -> pd.DataFrame | None:
    p = Path(ruta)
    if not p.exists():
        return None
    df = pd.read_csv(p, **LEER)
    for c in fechas:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
    return df


def csv(ruta: Path, fechas: tuple = ()) -> pd.DataFrame | None:
    return _csv(str(ruta), fechas, _firma(ruta))


def cargar() -> dict:
    carpeta = DATA / f"modelo_h{H}"
    est = DATA / "app" / "estado.json"
    return {
        "pred": csv(carpeta / "prediccion_proximo_dia.csv", ("fecha_ultimo_dato", "fecha_predicha")),
        "bt": csv(carpeta / "predicciones_diarias.csv", ("fecha_dato", "fecha_predicha")),
        "met": csv(carpeta / "metricas_backtest.csv"),
        "met_h1": csv(DATA / "modelo_h1" / "metricas_backtest.csv"),
        "imp": csv(DATA / "modelos" / "importancia_variables.csv"),
        "dataset": csv(DATA / "dataset_dolar_blue.csv", ("fecha",)),
        "estado": json.loads(est.read_text(encoding="utf-8")) if est.exists() else {},
    }


def fila_metrica(met: pd.DataFrame | None, periodo: str) -> pd.Series | None:
    if met is None:
        return None
    r = met[met["periodo"] == periodo]
    return r.iloc[0] if len(r) else None


def serie_blue(ds: pd.DataFrame | None) -> pd.DataFrame:
    """Precio de venta del blue en días hábiles (lunes a viernes)."""
    if ds is None:
        return pd.DataFrame(columns=["fecha", "blue_venta"])
    s = ds[ds["fecha"].dt.dayofweek < 5].dropna(subset=["blue_venta"])
    return s[["fecha", "blue_venta"]].sort_values("fecha").reset_index(drop=True)


def avisos_datos(d: dict) -> list[str]:
    """Avisos de calidad: dato viejo, MEP/CCL repetidos (hueco de la fuente) y avisos del pipeline."""
    avisos = []
    p = d["pred"].iloc[0]
    dias = (pd.Timestamp(datetime.now(ZONA).date()) - p["fecha_ultimo_dato"]).days
    if dias > DIAS_DATO_VIEJO:
        avisos.append(f"Los datos no se actualizan desde hace {dias} días: la proyección puede estar desactualizada.")
    ds = d["dataset"]
    if ds is not None:
        hab = ds[ds["fecha"].dt.dayofweek < 5]
        for col, nombre in (("dolar_mep", "MEP"), ("dolar_ccl", "CCL")):
            s = hab[col].dropna().tail(6) if col in hab else pd.Series(dtype=float)
            if len(s) == 6 and s.nunique() == 1:
                avisos.append(f"El dólar {nombre} repite el mismo valor ({pesos_md(s.iloc[-1], 2)}) en los últimos 6 días hábiles: "
                              "puede ser un hueco de la fuente de datos y la proyección podría ser menos precisa.")
    return avisos + list(d["estado"].get("avisos", []))


def texto_actualizado(d: dict) -> str:
    act = d["estado"].get("actualizado")
    return f"Actualizado el {datetime.fromisoformat(act):%d/%m/%Y a las %H:%M}" if act else ""


# ───────────────────────────── ESTILO ─────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@600;700&display=swap');
:root {{
  --fondo:{C['fondo']}; --panel:{C['panel']}; --panel2:{C['panel2']}; --borde:{C['borde']}; --texto:{C['texto']};
  --suave:{C['suave']}; --verde:{C['verde']}; --verde-osc:{C['verde_osc']}; --rojo:{C['rojo']}; --ambar:{C['ambar']};
}}
.stApp {{ background: radial-gradient(1200px 500px at 50% -120px, #16345a 0%, var(--fondo) 60%) fixed; }}
.stApp, .stApp p, .stApp li, .stApp label {{ font-family: 'Barlow', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1080px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
#MainMenu, [data-testid="stToolbar"], footer {{ visibility: hidden; }}

.barra {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
  padding: 10px 2px 14px; border-bottom: 1px solid var(--borde); margin-bottom: 18px; }}
.marca {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.15rem; color:var(--texto); letter-spacing:.2px; }}
.marca .logo {{ width:34px; height:34px; border-radius:9px; display:grid; place-items:center; font-weight:700;
  background: linear-gradient(135deg, var(--verde), var(--verde-osc)); color:#06231c; font-family:'Barlow Condensed', 'Barlow', sans-serif; font-size:1.25rem; }}
.marca small {{ display:block; font-weight:500; font-size:.78rem; color:var(--suave); letter-spacing:.3px; }}
.chip {{ font-size:.8rem; color:var(--suave); border:1px solid var(--borde); border-radius:999px; padding:5px 12px; background:rgba(255,255,255,.03); }}
.chip b {{ color: var(--texto); font-weight:600; }}

.titulo {{ font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size: clamp(2rem, 5vw, 2.9rem);
  color:var(--texto); line-height:1.05; margin: 6px 0 4px; }}
.titulo span {{ color: var(--verde); }}
.bajada {{ color: var(--suave); font-size: 1.02rem; margin-bottom: 20px; }}

.tarjetas {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap:18px; }}
.tarjeta {{ background: linear-gradient(180deg, var(--panel2), var(--panel)); border:1px solid var(--borde);
  border-radius:18px; padding:22px 24px 20px; position:relative; overflow:hidden; }}
.tarjeta.proy {{ border-color: rgba(25,195,156,.55); box-shadow: 0 0 0 1px rgba(25,195,156,.15), 0 18px 40px -18px rgba(25,195,156,.45); }}
.tarjeta.proy::before {{ content:""; position:absolute; inset:0 0 auto 0; height:4px; background: linear-gradient(90deg, var(--verde), #5fe0c0); }}
.tarjeta.alerta {{ border-color: rgba(255,93,108,.6); box-shadow: 0 18px 40px -18px rgba(255,93,108,.45); }}
.tarjeta.alerta::before {{ background: linear-gradient(90deg, var(--rojo), #ff9aa4); }}
.etiqueta {{ text-transform:uppercase; letter-spacing:1.4px; font-size:.78rem; font-weight:600; color:var(--suave); }}
.sub {{ color:var(--suave); font-size:.92rem; margin-top:2px; }}
.valor {{ font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size: clamp(3.2rem, 9vw, 4.6rem); line-height:1;
  color: var(--texto); margin: 14px 0 6px; font-variant-numeric: tabular-nums; }}
.tarjeta.proy .valor {{ color: var(--verde); }}
.tarjeta.alerta .valor {{ color: #ffd2d6; }}
.valor .moneda {{ font-size:.55em; vertical-align: .55em; margin-right:2px; opacity:.85; }}
.pie {{ color:var(--suave); font-size:.95rem; }}
.pie b {{ color: var(--texto); font-weight:600; }}
.delta {{ display:inline-block; font-weight:700; font-size:.95rem; padding:4px 10px; border-radius:8px; margin-right:6px; }}
.delta.sube {{ background: rgba(25,195,156,.15); color: var(--verde); }}
.delta.baja {{ background: rgba(255,93,108,.15); color: var(--rojo); }}
.delta.igual {{ background: rgba(149,167,192,.15); color: var(--suave); }}

.rango {{ margin-top:16px; padding-top:14px; border-top:1px dashed var(--borde); }}
.rango .cifras {{ display:flex; justify-content:space-between; font-weight:600; color:var(--texto); font-size:1.02rem; }}
.pista {{ position:relative; height:12px; border-radius:999px; background: rgba(255,255,255,.07); margin:10px 0 6px; }}
.pista .seg {{ position:absolute; top:0; bottom:0; border-radius:999px; background: linear-gradient(90deg, rgba(25,195,156,.35), rgba(25,195,156,.75)); }}
.tarjeta.alerta .pista .seg {{ background: linear-gradient(90deg, rgba(255,93,108,.35), rgba(255,93,108,.75)); }}
.pista .punto {{ position:absolute; top:-4px; width:20px; height:20px; margin-left:-10px; border-radius:50%;
  background: var(--texto); border: 4px solid var(--verde); box-shadow: 0 0 0 4px rgba(25,195,156,.2); }}
.tarjeta.alerta .pista .punto {{ border-color: var(--rojo); }}
.pista .hoy {{ position:absolute; top:-6px; bottom:-6px; width:2px; margin-left:-1px; background: var(--suave); }}
.spark {{ margin-top:10px; }}
.spark img {{ width:100%; height:64px; display:block; }}
.leyenda-rango {{ font-size:.82rem; color: var(--suave); }}

.aviso {{ display:flex; gap:14px; align-items:flex-start; border-radius:14px; padding:14px 18px; margin: 18px 0 6px; font-size:1rem; line-height:1.45; }}
.aviso .icono {{ font-size:1.35rem; line-height:1.1; }}
.aviso.ok {{ background: rgba(25,195,156,.10); border:1px solid rgba(25,195,156,.45); color:#cdf5ea; }}
.aviso.mal {{ background: rgba(255,93,108,.12); border:1px solid rgba(255,93,108,.55); color:#ffdfe2; }}
.aviso.dato {{ background: rgba(244,183,64,.10); border:1px solid rgba(244,183,64,.45); color:#fbe7bd; font-size:.92rem; margin:8px 0 0; padding:10px 16px; }}
.aviso b {{ color:#fff; }}

.seccion {{ font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:1.7rem; color:var(--texto); margin: 34px 0 2px; }}
.seccion-sub {{ color: var(--suave); margin-bottom: 8px; }}

.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:14px; margin: 8px 0 6px; }}
.kpi {{ background: var(--panel); border:1px solid var(--borde); border-radius:14px; padding:16px 18px; }}
.kpi .n {{ font-family:'Barlow Condensed', 'Barlow', sans-serif; font-weight:700; font-size:2.3rem; color: var(--verde); line-height:1.05; }}
.kpi .n.neutro {{ color: var(--texto); }}
.kpi .t {{ color: var(--suave); font-size:.92rem; margin-top:4px; }}
.pasos {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:14px; margin-top:10px; }}
.paso {{ background: var(--panel); border:1px solid var(--borde); border-radius:14px; padding:16px 18px; color: var(--texto); }}
.paso .num {{ width:30px; height:30px; border-radius:50%; display:grid; place-items:center; background: rgba(25,195,156,.15);
  color: var(--verde); font-weight:700; margin-bottom:8px; }}
.paso h4 {{ margin:0 0 6px; font-size:1.08rem; color: var(--texto); }}
.paso p {{ margin:0; color: var(--suave); font-size:.95rem; line-height:1.45; }}
.caja {{ background: var(--panel); border:1px solid var(--borde); border-radius:14px; padding:16px 20px; color: var(--texto);
  line-height:1.55; margin: 8px 0; }}
.caja p {{ margin: 0 0 8px; }}
.caja .muted {{ color: var(--suave); }}
.nota {{ color: var(--suave); font-size:.88rem; margin-top: 4px; }}

[data-testid="stPageLink"] a {{ background: linear-gradient(90deg, rgba(25,195,156,.18), rgba(25,195,156,.06)); border:1px solid rgba(25,195,156,.5);
  border-radius:14px; padding: 14px 18px; }}
[data-testid="stPageLink"] a:hover {{ background: rgba(25,195,156,.25); }}
[data-testid="stPageLink"] a, [data-testid="stPageLink"] a div, [data-testid="stPageLink"] a p {{ white-space: normal !important;
  overflow: visible !important; text-overflow: clip !important; height: auto !important; max-width: 100%; }}
[data-testid="stPageLink"] a p {{ font-size:1.05rem; font-weight:600; color: var(--texto); }}
.legal {{ color: var(--suave); font-size:.82rem; text-align:center; margin-top: 14px; line-height:1.5; }}
</style>
"""


def estilos() -> None:
    st.html(CSS)


def barra_superior(d: dict) -> None:
    p = d["pred"].iloc[0] if d["pred"] is not None else None
    chip = f"<span class='chip'>Último dato: <b>{p['fecha_ultimo_dato']:%d/%m/%Y}</b></span>" if p is not None else ""
    st.html(f"""<div class="barra">
      <div class="marca"><div class="logo">B</div><div>Pronóstico Blue<small>Proyecto académico de Machine Learning</small></div></div>
      {chip}</div>""")


def pie_legal(d: dict) -> None:
    act = texto_actualizado(d)
    st.html(f"""<div class="legal">{act + ' · ' if act else ''}Herramienta informativa de un proyecto académico. No es una recomendación
      de compra o venta.<br>Datos: MonedAPI, ArgentinaDatos, BCRA y Universidad Torcuato Di Tella. App versión {VERSION_APP}.</div>""")
