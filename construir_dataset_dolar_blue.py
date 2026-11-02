#!/usr/bin/env python3
"""
Construcción del dataset diario para predecir el dólar blue (Argentina).  -- versión 3.4 (API BCRA v4.0 + Merval empalmado + ETF ARGT + dólar MEP y CCL obligatorios)

Fuentes:
  - Dólar blue y oficial: MonedAPI y/o ArgentinaDatos según FUENTE_BLUE / FUENTE_OFICIAL (por defecto, ambos: MonedAPI)
  - Dólar MEP (bolsa) y CCL: MonedAPI (pares USD/BOLSA y USD/CCL) con ArgentinaDatos de respaldo. Son OBLIGATORIOS: el modelo
    siempre los usa. Cada par tiene su propia fecha de inicio: antes de su primer dato la columna queda VACÍA (no se rellena) y el
    modelo se entrena desde ese día. El log informa desde cuándo hay datos.
  - ArgentinaDatos (API pública): riesgo país, inflación mensual
  - BCRA, API Estadísticas Monetarias v4.0 (manual "Variables Monetarias v4.0"):
        reservas, BADLAR, plazo fijo 30 días
  - ARGT (ETF Global X MSCI Argentina, en USD, Yahoo): columna argt_var_pct. Es el índice "MSCI All Argentina 25/50", NO el MSCI Merval.
  - Merval: S&P Merval hasta el 16/10/2026 (Yahoo, ^MERV) + MSCI Merval desde el 19/10/2026 (raw/merval_msci.csv, a mano).
        Se empalman por VARIACIÓN diaria: la columna del dataset es merval_var_pct (%), no el nivel (la base cambió de ≈2,85 millones a 100.000).
  - Universidad Di Tella: ICG e ICC, desde los Excel cargados a mano en raw/ (ICG.xlsx, ICC.xlsx; no se descargan)

Decisión de diseño (diarias vs. mensuales)
  - La tabla final es DIARIA (el objetivo, el blue, es diario).
  - Cada variable se une con un "as-of join": a cada día se le asigna el último valor que ya era
    PÚBLICO ese día. Para las mensuales no se usa el mes de referencia sino la fecha estimada de
    publicación (PUBLICACION_MENSUAL), lo que evita el sesgo por anticipación (look-ahead).

FORMATO DE LOS CSV (importante al leerlos): separador de columnas ";", COMA como decimal y fechas DD/MM/AAAA.
    pandas : df = pd.read_csv("data/dataset_dolar_blue.csv", sep=";", decimal=",", encoding="utf-8-sig")
             df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
             (o simplemente  m.leer_dataset()  )
    R      : df <- read.csv2("data/dataset_dolar_blue.csv", fileEncoding = "UTF-8-sig")
             df$fecha <- as.Date(df$fecha, format = "%d/%m/%Y")
    Excel  : Datos > Desde un archivo de texto/CSV > delimitador "punto y coma" (con Excel en español abre directo).
    Power BI: Obtener datos > Texto/CSV > delimitador "Punto y coma", origen UTF-8; si los decimales salen mal,
             Transformar datos > Usar configuración regional > Español (Argentina).
Cada vez que el script guarda el dataset también escribe data/LEEME_formato_dataset.txt con estas mismas instrucciones.

Uso en Colab:
    !pip install -q yfinance openpyxl xlrd
    import construir_dataset_dolar_blue as m
    df = m.main()                      # construye todo el dataset
    df = m.actualizar_icc_icg()        # solo vuelve a leer raw/ICC.xlsx y raw/ICG.xlsx y actualiza las columnas icc/icg del CSV
"""
from __future__ import annotations

import logging
import re
import shutil
import time
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("dataset")
log.setLevel(logging.INFO)        # en Colab el logging ya viene configurado y basicConfig no hace nada

VERSION = "3.4"               # se imprime al ejecutar main(): sirve para confirmar qué código está corriendo

# ───────────────────────────── CONFIGURACIÓN ─────────────────────────────
FECHA_INICIO = "2016-01-01"
FECHA_FIN = date.today().isoformat()

DIR_RAW = Path("data/raw")        # carpeta con los Excel cargados a mano (también se busca en raw/)
DIR_OUT = Path("data")

# Carpeta EXTRA donde buscar ICC.xlsx / ICG.xlsx (opcional). Ej. en Colab con Drive:
#   DIR_EXCEL_EXTRA = "/content/drive/MyDrive/proyecto_dolar_blue/raw"
# Aunque la dejes en None, el script también busca en: raw/, data/raw/, data/ y la carpeta actual,
# tanto relativas a la carpeta de trabajo como a la carpeta donde está este .py.
DIR_EXCEL_EXTRA = None

# Formato de los CSV que genera el script: coma decimal y ";" como separador de columnas
# (con coma decimal no se puede usar "," para separar columnas; ";" es el estándar en Excel en español).
CSV_SEP = ";"
CSV_DECIMAL = ","
CSV_FECHA = "%d/%m/%Y"      # fechas en los CSV: día/mes/año (ej. 23/10/2026)

ARGENTINADATOS = "https://api.argentinadatos.com/v1"

# MonedAPI (https://monedapi.ar/docs): cotizaciones con compra/venta y decimales. Sin clave según la documentación consultada.
MONEDAPI = "https://monedapi.ar/api/v2"
MONEDAPI_LIMITE = 500              # máximo de puntos por pedido (doc: por defecto 100, tope 500)
MONEDAPI_VENTANA_DIAS = 180        # rango por pedido; si llega al tope de puntos, se divide a la mitad
MONEDAPI_PAUSA_S = 0.7             # doc: más de 20 pedidos en 10 s por IP → error 429

# De dónde sale cada tipo de cambio: "monedapi" | "argentinadatos" | "promedio" (media diaria de ambas fuentes).
# Si la fuente elegida falla, el script usa la otra y lo avisa en el log.
FUENTE_BLUE = "monedapi"
FUENTE_OFICIAL = "monedapi"
FUENTE_MEP_CCL = "monedapi"
# Nombre de cada dólar en cada fuente (MonedAPI: pares USD/BOLSA y USD/CCL según su documentación;
# ArgentinaDatos: "bolsa" y "contadoconliqui", no verificados en su documentación → si fallan, el log lo informa)
CASAS_POR_FUENTE = {
    "mep": {"monedapi": "bolsa", "argentinadatos": "bolsa"},
    "ccl": {"monedapi": "ccl", "argentinadatos": "contadoconliqui"},
}

# BCRA v4.0 (manual: GET /estadisticas/v4.0/monetarias y /monetarias/{idVariable}?desde&hasta&limit&offset)
BCRA_BASE = "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias"
BCRA_VERIFY_SSL = True            # poner False si falla el certificado del BCRA
BCRA_LIMIT_CATALOGO = 1000        # default del manual para el listado
BCRA_LIMIT_SERIE = 3000           # máximo permitido por el manual para las series
MAX_PAGINAS = 50                  # tope de seguridad para la paginación
MARGEN_INICIAL_DIAS = 45          # se piden datos desde antes de FECHA_INICIO para que las primeras filas no queden vacías

# Rezago de publicación de las series DIARIAS (días). Supuesto: el BCRA publica con 1 día de demora.
REZAGO_DIARIO = {"mercado": 0, "bcra": 1}

# Cuándo queda disponible cada serie MENSUAL: (meses_después_del_mes_de_referencia, día_del_mes)
# SUPUESTOS CONSERVADORES: ajustar con el calendario de INDEC y el cronograma de difusión de UTDT.
PUBLICACION_MENSUAL = {
    "inflacion_mensual_pct": (1, 15),   # INDEC publica alrededor de mediados del mes siguiente
    # ICG e ICC: el valor de un mes se asigna a los días de ESE mismo mes (0 meses, desde el día 1).
    # Si todavía no existe el valor de un mes (p. ej. octubre), se arrastra el ÚLTIMO valor disponible.
    # Para evitar el sesgo de anticipación (usar el dato recién cuando se publica) poné (1, 1): día 1 del mes siguiente.
    "icg": (0, 1),
    "icc": (0, 1),
}

# Variables del BCRA: se buscan en el catálogo v4 por texto de "descripcion" + "unidadExpresion"
# (en v4 la unidad, p. ej. "% n.a.", viene en un campo aparte), solo entre series diarias (periodicidad D).
#   incluye : todos los términos deben aparecer (un tuple = alguno de ellos)
#   excluye : ningún término puede aparecer
#   prefiere: desempata a favor de quien contenga más de estos términos (luego, menor idVariable)
# Textos sin tildes y en minúsculas. El catálogo completo se guarda en data/bcra_catalogo_v4.csv para auditarlo.
SPECS_BCRA = {
    "reservas_millones_usd": {
        "incluye": ["reservas internacionales"],
        "excluye": ["variacion", "neta", "compra"], "prefiere": ["usd", "dolar"]},
    "tasa_plazo_fijo_minorista_tna": {
        "incluye": ["30 dias", ("depositos", "plazo fijo")],
        "excluye": [" uva", "e.a.", "efectiva", "prestamo", "adelanto", "mayorist"], "prefiere": ["n.a."]},
    "tasa_badlar_pct": {
        "incluye": ["badlar"],
        "excluye": ["e.a.", "efectiva", "publico"], "prefiere": ["privados", "n.a."]},
}
# Si la búsqueda automática elige mal, forzá el ID acá. Ej: {"tasa_badlar_pct": 7}
BCRA_IDS_OVERRIDE: dict[str, int] = {}

# ICG e ICC (Di Tella) se cargan A MANO como Excel en la carpeta raw/: el script NO los descarga de internet.
# "rango" = valores válidos del índice; sirve para detectar si se eligió la columna equivocada.
UTDT = {
    "icg": {"rango": (0, 5)},
    "icc": {"rango": (0, 100)},
}

# MERVAL: empalme S&P Merval (hasta MSCI_BASE_FECHA) → MSCI Merval (desde el día hábil siguiente).
# Dato de las fuentes consultadas (Dolarito, Ámbito, Perfil): el MSCI Merval arrancó con base 100.000 puntos el 16/10/2026
# y debutó en el mercado el 19/10/2026 (cierre 19/10 = 98.214,70). No se verificó que MSCI publique una serie histórica
# hacia atrás, por eso el empalme se hace con la VARIACIÓN diaria (continua aunque cambie la base).
# Yahoo (^MERV) quedó congelado desde esa fecha: los cierres del MSCI Merval se cargan a mano en raw/merval_msci.csv
# (columnas: fecha;cierre, fecha DD/MM/AAAA, coma decimal; un renglón por día hábil desde el 19/10/2026).
MSCI_BASE_FECHA = "2026-10-16"
MSCI_BASE_VALOR = 100000.0
# ARGT = ETF Global X MSCI Argentina (NYSE Arca, en USD). OJO: replica el "MSCI All Argentina 25/50", que NO es el MSCI Merval
# (otra composición: MercadoLibre ≈23 %, acciones/ADR que cotizan en EE.UU.; y está en dólares). Se usa como variable
# adicional automática, no como reemplazo del Merval.
ARGT_TICKER = "ARGT"
MSCI_RANGO = (20_000, 1_000_000)     # valores plausibles del MSCI Merval; detecta si se pegó el nivel viejo (≈2,8 millones)

# Nombres exactos de los Excel cargados a mano (se buscan en raw/, data/raw/, data/ y la carpeta actual)
ARCHIVOS_LOCALES = {"icg": ("ICG.xlsx", "icg.xlsx"), "icc": ("ICC.xlsx", "icc.xlsx")}

COLUMNAS_BASE = ["blue_venta", "dolar_oficial", "tasa_plazo_fijo_minorista_tna", "inflacion_mensual_pct"]
# Columnas que ya no forman parte del dataset; si un CSV viejo las trae, se quitan al actualizar.
#  - spread_blue_vs_plazo_fijo: decisión del usuario.
#  - tasa_politica_monetaria_tna: la TPM dejó de actualizarse (último cambio 01/02/2025 = 29%); luego era un valor arrastrado.
#  - tasa_tm20_tna: redundante con la BADLAR (corr. 0,998 en nivel, 0,92 en variaciones) y sin datos antes del 03/01/2017.
#  - merval_cierre: el nivel del Merval no es comparable tras el cambio S&P → MSCI (19/10/2026, base 100.000 vs ≈2,85 millones);
#    se reemplaza por merval_var_pct (variación diaria %), que sí se puede empalmar.
COLUMNAS_ELIMINADAS = ["spread_blue_vs_plazo_fijo", "tasa_politica_monetaria_tna", "tasa_tm20_tna", "merval_cierre"]
ORDEN_COLUMNAS = [
    "fecha", "blue_venta", "dolar_oficial", "dolar_mep", "dolar_ccl", "brecha_pct",
    "tasa_real_mensual", "tasa_plazo_fijo_minorista_tna",
    "tasa_badlar_pct",
    "inflacion_mensual_pct", "reservas_millones_usd", "merval_var_pct", "argt_var_pct", "riesgo_pais_pb",
    "icg", "icc",
]


# ───────────────────────────── UTILIDADES ─────────────────────────────
def _norm(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _ns(serie) -> pd.Series:
    """Fechas siempre en datetime64[ns]: merge_asof exige la misma resolución en ambos lados (pandas 2 vs 3)."""
    return pd.to_datetime(serie).astype("datetime64[ns]")


def _get(url, params=None, verify=True, reintentos=3, timeout=30):
    """GET con reintentos. Los errores 4xx (salvo 429) no se reintentan: son errores de la consulta."""
    ultimo = None
    for i in range(reintentos):
        try:
            r = requests.get(url, params=params, timeout=timeout, verify=verify,
                             headers={"User-Agent": "Mozilla/5.0 (proyecto-academico)"})
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code is not None and 400 <= code < 500 and code != 429:
                cuerpo = (getattr(e.response, "text", "") or "")[:200]
                raise RuntimeError(f"HTTP {code} en {url} | {cuerpo}") from e
            ultimo = e
        except Exception as e:  # noqa: BLE001
            ultimo = e
        time.sleep(2 * (i + 1))
    pista = " (si es un error de certificado, probá BCRA_VERIFY_SSL = False)" if "SSL" in str(ultimo).upper() else ""
    raise RuntimeError(f"Falló la consulta a {url}: {ultimo}{pista}")


def _desde_con_margen() -> str:
    return (pd.Timestamp(FECHA_INICIO) - pd.Timedelta(days=MARGEN_INICIAL_DIAS)).date().isoformat()


def _diaria(df: pd.DataFrame, col: str, rezago: int) -> pd.DataFrame:
    """Serie diaria → formato as-of (disponible_desde, valor)."""
    out = df[["fecha", col]].dropna().drop_duplicates("fecha", keep="last").copy()
    out["disponible_desde"] = _ns(out["fecha"] + pd.Timedelta(days=rezago))
    return out[["disponible_desde", col]].sort_values("disponible_desde")


def _mensual(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Serie mensual (periodo, valor) → formato as-of usando la fecha estimada de publicación."""
    meses, dia = PUBLICACION_MENSUAL[col]
    out = df[["periodo", col]].dropna().drop_duplicates("periodo", keep="last").copy()
    inicio_mes = out["periodo"].dt.to_timestamp()
    out["disponible_desde"] = _ns(inicio_mes + pd.DateOffset(months=meses) + pd.Timedelta(days=dia - 1))
    return out[["disponible_desde", col]].sort_values("disponible_desde")


# ───────────────────────────── ARGENTINADATOS ─────────────────────────────
def dolar_argentinadatos(casa: str, col: str) -> pd.DataFrame:
    df = pd.DataFrame(_get(f"{ARGENTINADATOS}/cotizaciones/dolares/{casa}").json())
    if not {"fecha", "venta"} <= set(df.columns):
        raise ValueError(f"dólar {casa}: columnas inesperadas {list(df.columns)}")
    df["fecha"] = _ns(df["fecha"])
    return df[["fecha", "venta"]].rename(columns={"venta": col})


# ───────────────────────────── MONEDAPI ─────────────────────────────
def _monedapi_get(ruta: str, params: dict | None = None):
    """GET a MonedAPI respetando el límite de la doc (20 pedidos/10 s): pausa fija y, ante 429/503, espera Retry-After."""
    ultimo = ""
    for _ in range(6):
        time.sleep(MONEDAPI_PAUSA_S)
        r = requests.get(f"{MONEDAPI}{ruta}", params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0 (proyecto-academico)"})
        if r.status_code in (429, 503):
            try:
                espera = float((getattr(r, "headers", None) or {}).get("Retry-After", 10))
            except (TypeError, ValueError):
                espera = 10.0
            ultimo = f"HTTP {r.status_code}"
            time.sleep(espera + 1)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} en MonedAPI {ruta} {params} | {(getattr(r, 'text', '') or '')[:200]}")
        return r.json()
    raise RuntimeError(f"MonedAPI no respondió tras varios reintentos ({ultimo}) en {ruta}")


def _monedapi_historia(origen: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> list[dict]:
    """
    Puntos {buy, sell, createdAt} de /history/usd/{origen}. Las fechas del pedido son días UTC (no argentinos).
    Pide ventanas de MONEDAPI_VENTANA_DIAS; si una respuesta llega al tope de puntos, la divide en dos.
    """
    puntos: list[dict] = []

    def pedir(a: pd.Timestamp, b: pd.Timestamp) -> None:
        datos = _monedapi_get(f"/history/usd/{origen}", {"date_from": a.date().isoformat(), "date_to": b.date().isoformat(),
                                                        "limit": MONEDAPI_LIMITE})
        if not isinstance(datos, list):
            raise ValueError(f"MonedAPI history/{origen}: respuesta inesperada {str(datos)[:150]}")
        if len(datos) >= MONEDAPI_LIMITE and a < b:
            mitad = a + pd.Timedelta(days=(b - a).days // 2)
            pedir(a, mitad)
            pedir(mitad + pd.Timedelta(days=1), b)
            return
        if len(datos) >= MONEDAPI_LIMITE:
            log.warning("MonedAPI %s: el día %s llegó al tope de %d puntos; podrían faltar cambios.", origen, a.date(), MONEDAPI_LIMITE)
        puntos.extend(datos)

    a = desde
    while a <= hasta:
        b = min(a + pd.Timedelta(days=MONEDAPI_VENTANA_DIAS - 1), hasta)
        pedir(a, b)
        a = b + pd.Timedelta(days=1)
    return puntos


def dolar_monedapi(origen: str, col: str) -> pd.DataFrame:
    """
    Serie diaria (calendario completo) de la VENTA en MonedAPI: último valor de cada día argentino, y se arrastra
    en los días sin cambios (la API devuelve un punto solo cuando el valor cambia).
    """
    inicio = pd.Timestamp(_desde_con_margen())
    fin = pd.Timestamp(FECHA_FIN)
    # valor vigente al inicio (si en el rango inicial no hubo cambios, el histórico devuelve [])
    semilla = _monedapi_get(f"/date/{inicio.date().isoformat()}/usd/{origen}")
    puntos = _monedapi_historia(origen, inicio - pd.Timedelta(days=1), fin + pd.Timedelta(days=1))
    if isinstance(semilla, dict) and semilla.get("sell") is not None and semilla.get("quotedAt"):
        puntos.append({"buy": semilla.get("buy"), "sell": semilla["sell"], "createdAt": semilla["quotedAt"]})
    if not puntos:
        raise ValueError(f"MonedAPI usd/{origen}: sin datos entre {inicio.date()} y {fin.date()}")
    df = pd.DataFrame(puntos).drop_duplicates("createdAt")
    if not {"sell", "createdAt"} <= set(df.columns):
        raise ValueError(f"MonedAPI usd/{origen}: columnas inesperadas {list(df.columns)}")
    df["sell"] = pd.to_numeric(df["sell"], errors="coerce")
    t = pd.to_datetime(df["createdAt"], utc=True).dt.tz_convert("America/Argentina/Buenos_Aires")
    df = df.assign(t=t, dia=_ns(t.dt.tz_localize(None).dt.normalize())).dropna(subset=["sell"]).sort_values("t")
    if (df["sell"] <= 0).any():
        raise ValueError(f"MonedAPI usd/{origen}: hay valores de venta <= 0")
    diaria = df.groupby("dia")["sell"].last()
    calendario = pd.date_range(diaria.index.min(), max(fin, diaria.index.max()), freq="D")
    serie = diaria.reindex(calendario).ffill()
    serie = serie[serie.index >= inicio]
    log.info("MonedAPI usd/%s: %d puntos → %d días (%s → %s)", origen, len(df), len(serie), serie.index.min().date(), serie.index.max().date())
    return pd.DataFrame({"fecha": _ns(pd.Series(serie.index)), col: serie.values})


def obtener_dolar(casa: str, col: str, fuente: str) -> pd.DataFrame:
    """Serie diaria del dólar según `fuente` ("monedapi", "argentinadatos" o "promedio"); si una falla usa la otra."""
    nombres = CASAS_POR_FUENTE.get(casa, {})
    fuentes = {"monedapi": lambda: dolar_monedapi(nombres.get("monedapi", casa), col),
               "argentinadatos": lambda: dolar_argentinadatos(nombres.get("argentinadatos", casa), col)}
    if fuente not in fuentes and fuente != "promedio":
        raise ValueError(f"fuente '{fuente}' inválida: usá 'monedapi', 'argentinadatos' o 'promedio'")
    orden = list(fuentes) if fuente == "promedio" else [fuente] + [f for f in fuentes if f != fuente]
    ok: dict[str, pd.DataFrame] = {}
    for f in orden:
        try:
            ok[f] = fuentes[f]()
        except Exception as e:  # noqa: BLE001
            log.warning("%s: falló la fuente %s (%s)", col, f, e)
        if fuente != "promedio" and ok:
            break
    if not ok:
        raise RuntimeError(f"{col}: no se pudo obtener de ninguna fuente")
    if fuente != "promedio":
        usada = next(iter(ok))
        if usada != fuente:
            log.warning("%s: se usó %s en lugar de %s", col, usada, fuente)
        return ok[usada]
    if len(ok) == 1:
        log.warning("%s: 'promedio' con una sola fuente disponible (%s)", col, next(iter(ok)))
        return next(iter(ok.values()))
    m = ok["monedapi"].merge(ok["argentinadatos"], on="fecha", how="outer", suffixes=("_m", "_a")).sort_values("fecha")
    media = m[[f"{col}_m", f"{col}_a"]].mean(axis=1, skipna=True)
    return pd.DataFrame({"fecha": m["fecha"].values, col: media.values}).dropna()


def comparar_fuentes_blue(casa: str = "blue") -> pd.DataFrame:
    """Baja el mismo dólar de MonedAPI y ArgentinaDatos y los compara día por día (para decidir qué fuente usar)."""
    nombres = CASAS_POR_FUENTE.get(casa, {})
    m = dolar_monedapi(nombres.get("monedapi", casa), "monedapi")
    a = dolar_argentinadatos(nombres.get("argentinadatos", casa), "argentinadatos")
    j = m.merge(a, on="fecha", how="inner").set_index("fecha")
    j = j[j.index >= pd.Timestamp(FECHA_INICIO)]
    j["dif_pct"] = (j["monedapi"] / j["argentinadatos"] - 1) * 100
    hab = j[j.index.dayofweek < 5]
    print(f"Comparación {casa}: {len(j)} días en común ({j.index.min().date()} → {j.index.max().date()})")
    print(f"  Días con el mismo valor: {(j['dif_pct'].abs() < 1e-9).mean() * 100:.0f}% | diferencia absoluta media {j['dif_pct'].abs().mean():.2f}% | "
          f"p95 {j['dif_pct'].abs().quantile(.95):.2f}% | máx {j['dif_pct'].abs().max():.2f}% ({j['dif_pct'].abs().idxmax().date()})")
    for nombre in ("monedapi", "argentinadatos"):
        x = j[nombre]; hx = hab[nombre]
        print(f"  {nombre:<15}: valores enteros {(x % 1 == 0).mean() * 100:3.0f}% | días hábiles sin cambio {(hx.diff() == 0).mean() * 100:3.0f}%")
    print("  Por año (diferencia absoluta media %):", j["dif_pct"].abs().groupby(j.index.year).mean().round(2).to_dict())
    return j


def serie_fecha_valor(ruta: str, col: str) -> pd.DataFrame:
    df = pd.DataFrame(_get(f"{ARGENTINADATOS}{ruta}").json())
    if not {"fecha", "valor"} <= set(df.columns):
        raise ValueError(f"{ruta}: columnas inesperadas {list(df.columns)}")
    df["fecha"] = _ns(df["fecha"])
    return df[["fecha", "valor"]].rename(columns={"valor": col})


# ───────────────────────────── BCRA v4.0 ─────────────────────────────
def _paginar_bcra(url: str, params: dict, limit: int, contar):
    """
    Pide páginas con offset/limit hasta que una página venga incompleta (< limit filas).
    No depende de metadata.resultset.count para cortar; solo se usa para advertir si no coincide.
    `contar(json)` devuelve cuántas filas trajo la página. Devuelve (lista de JSON, count informado).
    """
    paginas, offset, count, previo = [], 0, None, None
    for _ in range(MAX_PAGINAS):
        j = _get(url, params={**params, "limit": limit, "offset": offset}, verify=BCRA_VERIFY_SSL).json()
        if j.get("status", 200) != 200:
            raise RuntimeError(f"BCRA respondió status {j.get('status')}: {j.get('errorMessages')}")
        if count is None:
            count = (j.get("metadata") or {}).get("resultset", {}).get("count")
        if previo is not None and j.get("results") == previo:
            log.warning("BCRA: la página con offset=%s repite la anterior; corto la paginación", offset)
            break
        previo = j.get("results")
        n = contar(j)
        paginas.append(j)
        if n < limit:
            break
        offset += n
    else:
        log.warning("BCRA: se alcanzó MAX_PAGINAS=%d; los datos pueden estar incompletos", MAX_PAGINAS)
    return paginas, count


def bcra_listar_variables() -> pd.DataFrame:
    """Catálogo completo de variables (v4 tiene más de 1000, hay que paginar)."""
    paginas, count = _paginar_bcra(BCRA_BASE, {}, BCRA_LIMIT_CATALOGO, lambda j: len(j.get("results", [])))
    df = pd.DataFrame([r for j in paginas for r in j.get("results", [])])
    if df.empty or not {"idVariable", "descripcion"} <= set(df.columns):
        raise RuntimeError(f"Catálogo BCRA vacío o con columnas inesperadas: {list(df.columns)}")
    for c in ("unidadExpresion", "periodicidad"):
        if c not in df.columns:
            df[c] = ""
    df = df.drop_duplicates("idVariable")
    if count is not None and len(df) != count:
        log.warning("BCRA: el catálogo informa %s variables y se leyeron %d", count, len(df))
    log.info("BCRA: catálogo leído (%d variables, %d página/s)", len(df), len(paginas))
    return df


def bcra_resolver_ids(catalogo: pd.DataFrame) -> dict[str, int]:
    texto = (catalogo["descripcion"].fillna("") + " " + catalogo["unidadExpresion"].fillna("")).map(_norm)
    diaria = catalogo["periodicidad"].fillna("").astype(str).str.upper().str[:1] == "D"
    ids = dict(BCRA_IDS_OVERRIDE)
    for col, spec in SPECS_BCRA.items():
        if col in ids:
            continue
        mask = diaria.copy()
        for t in spec["incluye"]:
            alternativas = (t,) if isinstance(t, str) else t
            mask &= texto.map(lambda s, a=alternativas: any(x in s for x in a))
        for t in spec["excluye"]:
            mask &= ~texto.map(lambda s, t=t: t in s)
        cand = catalogo[mask].copy()
        if cand.empty:
            log.warning("BCRA: no encontré una serie diaria para '%s'. Quedará vacía; buscala en "
                        "data/bcra_catalogo_v4.csv y forzá el ID en BCRA_IDS_OVERRIDE.", col)
            continue
        cand["_score"] = texto[mask].map(lambda s: sum(p in s for p in spec["prefiere"]))
        cand = cand.sort_values(["_score", "idVariable"], ascending=[False, True])
        if len(cand) > 1:
            log.warning("BCRA: '%s' tiene %d candidatas, uso la primera. Revisar:\n%s", col, len(cand),
                        cand[["idVariable", "descripcion", "unidadExpresion"]].to_string(index=False))
        elegido = cand.iloc[0]
        ids[col] = int(elegido["idVariable"])
        log.info("BCRA: %s → id %s | %s [%s]", col, ids[col], elegido["descripcion"], elegido["unidadExpresion"])
    return ids


def _filas_detalle(j: dict) -> list[dict]:
    filas = []
    for r in j.get("results", []):
        filas += r["detalle"] if "detalle" in r else [r]   # v4: results[{idVariable, detalle:[{fecha, valor}]}]
    return filas


def bcra_serie(id_var: int, col: str) -> pd.DataFrame:
    paginas, count = _paginar_bcra(f"{BCRA_BASE}/{id_var}", {"desde": _desde_con_margen(), "hasta": FECHA_FIN},
                                   BCRA_LIMIT_SERIE, lambda j: len(_filas_detalle(j)))
    filas = [f for j in paginas for f in _filas_detalle(j)]
    if not filas:
        raise RuntimeError(f"id {id_var}: sin datos entre {_desde_con_margen()} y {FECHA_FIN}")
    if count is not None and len(filas) != count:
        log.warning("BCRA id %s: informa %s registros y se leyeron %d", id_var, count, len(filas))
    df = pd.DataFrame(filas)
    df["fecha"] = _ns(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df[["fecha", "valor"]].rename(columns={"valor": col}).sort_values("fecha")   # v4 devuelve fechas descendentes


# ───────────────────────────── MERVAL ─────────────────────────────
def merval_sp() -> pd.DataFrame:
    """Cierres del S&P Merval desde Yahoo (^MERV). Desde el 16/10/2026 Yahoo no se actualiza (cambio a MSCI)."""
    import yfinance as yf
    h = yf.Ticker("^MERV").history(start=_desde_con_margen(), end=FECHA_FIN, auto_adjust=False)
    if h.empty:
        raise RuntimeError("yfinance no devolvió datos para ^MERV")
    idx = h.index.tz_localize(None) if h.index.tz is not None else h.index
    return pd.DataFrame({"fecha": _ns(idx.normalize()), "merval_nivel": h["Close"].values})


def argt() -> pd.DataFrame:
    """Cierres del ETF Global X MSCI Argentina (ARGT) desde Yahoo, en USD. Serie continua desde 2011, sin quiebres."""
    import yfinance as yf
    h = yf.Ticker(ARGT_TICKER).history(start=_desde_con_margen(), end=FECHA_FIN, auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"yfinance no devolvió datos para {ARGT_TICKER}")
    idx = h.index.tz_localize(None) if h.index.tz is not None else h.index
    return pd.DataFrame({"fecha": _ns(idx.normalize()), "argt_nivel": h["Close"].values})


def leer_merval_msci() -> pd.DataFrame | None:
    """
    Lee raw/merval_msci.csv (o .xlsx): cierres diarios del MSCI Merval cargados a mano (columnas fecha, cierre).
    Devuelve (fecha, merval_nivel) o None si el archivo no existe. Valida el rango para detectar niveles del índice viejo.
    """
    hallados, carpetas = _buscar_archivos_locales("merval_msci", (".csv", ".xlsx", ".xls", ".xlsm"))
    if not hallados:
        return None
    ruta = max(hallados, key=lambda p: p.stat().st_mtime)
    if ruta.suffix.lower() == ".csv":
        df = _leer_csv(ruta)
    else:
        df = pd.read_excel(ruta)
        df.columns = [str(c).strip().lower() for c in df.columns]
    if "fecha" not in df.columns:
        raise ValueError(f"{ruta.name}: falta la columna 'fecha'. Columnas halladas: {list(df.columns)}")
    cols = [c for c in df.columns if c != "fecha"]
    if not cols:
        raise ValueError(f"{ruta.name}: falta la columna con el cierre (ej. 'cierre').")
    val = pd.to_numeric(df[cols[0]], errors="coerce")
    out = pd.DataFrame({"fecha": _ns(pd.to_datetime(df["fecha"], dayfirst=True)), "merval_nivel": val}).dropna()
    out = out.drop_duplicates("fecha", keep="last").sort_values("fecha")
    lo, hi = MSCI_RANGO
    if not out["merval_nivel"].between(lo, hi).all():
        raise ValueError(f"{ruta.name}: hay cierres fuera de [{lo:,}; {hi:,}]. El MSCI Merval arrancó en 100.000; "
                         "probablemente se pegó el nivel del S&P Merval (≈2,8 millones) o hay un error de tipeo.")
    log.info("MSCI Merval: %s, %d cierres (%s → %s)", ruta.name, len(out), out["fecha"].min().date(), out["fecha"].max().date())
    return out


def merval() -> pd.DataFrame:
    """
    Nivel EMPALMADO del Merval, en base MSCI (base = MSCI_BASE_VALOR en MSCI_BASE_FECHA):
      · hasta la fecha base: S&P Merval multiplicado por (MSCI_BASE_VALOR / cierre del S&P en la fecha base);
      · después: cierres del MSCI Merval cargados a mano.
    Las variaciones diarias de esta serie son las reales de cada índice; solo el día de empalme usa el nivel base común.
    El nivel empalmado es auxiliar: al dataset va solo merval_var_pct.
    """
    base = pd.Timestamp(MSCI_BASE_FECHA)
    sp = merval_sp()
    sp_hasta = sp[sp["fecha"] <= base]
    if sp_hasta.empty:
        raise RuntimeError("Yahoo no trae datos del S&P Merval hasta la fecha base del empalme")
    ultimo_sp = sp_hasta.iloc[-1]
    if (base - ultimo_sp["fecha"]).days > 5:
        log.warning("Merval: el último cierre del S&P en Yahoo es del %s, lejos de la fecha base %s; revisá el empalme.",
                    ultimo_sp["fecha"].date(), base.date())
    factor = MSCI_BASE_VALOR / ultimo_sp["merval_nivel"]
    log.info("Merval: empalme en %s | S&P = %s ↔ MSCI = %s | factor %.6f",
             ultimo_sp["fecha"].date(), f"{ultimo_sp['merval_nivel']:,.2f}", f"{MSCI_BASE_VALOR:,.0f}", factor)
    parte_sp = sp_hasta.assign(merval_nivel=sp_hasta["merval_nivel"] * factor)

    msci = leer_merval_msci()
    if msci is None:
        log.warning("Merval: no encontré raw/merval_msci.csv. Sin él, merval_var_pct queda VACÍA desde el %s "
                    "(Yahoo no actualiza el índice nuevo). Cargá los cierres del MSCI Merval (fecha;cierre).",
                    (base + pd.Timedelta(days=1)).date())
        return parte_sp
    antes = msci[msci["fecha"] <= base]
    if len(antes):
        log.warning("Merval: raw/merval_msci.csv tiene %d fechas <= %s; se ignoran (ahí rige el S&P Merval).",
                    len(antes), base.date())
    return pd.concat([parte_sp, msci[msci["fecha"] > base]], ignore_index=True)


# ───────────────────────────── DI TELLA (ICG / ICC) ─────────────────────────────
_MESES = {"ene": 1, "jan": 1, "feb": 2, "mar": 3, "abr": 4, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "ago": 8, "aug": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12, "dec": 12}
_FORMATOS_FECHA = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m", "%Y/%m/%d", "%d/%m/%Y", "%m/%Y", "%Y%m")


def _fechas_texto_mes(s: pd.Series) -> pd.Series:
    """'mar-01', 'ene 2001', 'Mar-2001' → primer día del mes."""
    ext = s.str.extract(r"^([a-zñ]{3})[a-zñ]*[\s./-]+(\d{2}|\d{4})$")
    # float64 (no tipos "nullable"): np.where falla con pd.NA en pandas 3
    mes = pd.to_numeric(ext[0].map(_MESES), errors="coerce").astype("float64")
    anio = pd.to_numeric(ext[1], errors="coerce").astype("float64")
    anio = anio.where(anio >= 100, np.where(anio <= 69, 2000 + anio, 1900 + anio))
    return pd.to_datetime(pd.DataFrame({"year": anio, "month": mes, "day": 1}), errors="coerce")


def _mensual_valida(f: pd.Series) -> bool:
    """Una columna de fechas mensuales válida: ≥24 fechas entre 2000 y 2100, un registro por mes, ordenada."""
    v = f.dropna()
    v = v[(v >= "2000-01-01") & (v <= "2100-01-01")]
    if len(v) < 24 or v.dt.to_period("M").nunique() < 0.95 * len(v):
        return False
    return v.is_monotonic_increasing or v.is_monotonic_decreasing


def _fechas_anio_m_texto(s: pd.Series) -> pd.Series:
    """'2001m03', '2001.3', '2001/03' → primer día del mes."""
    ext = s.str.extract(r"^(\d{4})\s*[m./-]\s*(\d{1,2})$")
    anio = pd.to_numeric(ext[0], errors="coerce").astype("float64")
    mes = pd.to_numeric(ext[1], errors="coerce").astype("float64")
    return pd.to_datetime(pd.DataFrame({"year": anio, "month": mes, "day": 1}), errors="coerce")


def _leer_fechas(col: pd.Series) -> pd.Series | None:
    """
    Convierte una columna a fechas SIN adivinar: solo formatos explícitos, fechas reales de Excel,
    números de serie de Excel o enteros AAAAMM. Adivinar con dateutil convertiría 'mar-01' en
    "1 de marzo de este año" sin avisar.
    """
    if pd.api.types.is_datetime64_any_dtype(col):
        return col if _mensual_valida(col) else None
    candidatos = []
    num = pd.to_numeric(col, errors="coerce").astype("float64")
    if num.notna().sum() >= 24:
        serial = num[num.between(30000, 80000)]                             # días desde 1899-12-30
        dias = pd.to_timedelta(serial.astype("int64"), unit="D")            # solo valores válidos (con NaN, pandas 2.2 falla a veces)
        candidatos.append((pd.Timestamp("1899-12-30") + dias).reindex(num.index).astype("datetime64[ns]"))
        aaaamm = num.where(num.between(190001, 210012) & (num % 100).between(1, 12))
        candidatos.append(pd.to_datetime(aaaamm.astype("Int64").astype("string"), format="%Y%m", errors="coerce"))
    if not pd.api.types.is_numeric_dtype(col):
        s = col.astype("string").str.strip().str.lower()
        candidatos += [_fechas_texto_mes(s), _fechas_anio_m_texto(s)]
        candidatos += [pd.to_datetime(s, format=f, errors="coerce") for f in _FORMATOS_FECHA]
    validos = [c for c in candidatos if _mensual_valida(c)]
    return max(validos, key=lambda c: c.notna().sum()) if validos else None


def _fechas_anio_mes_separados(raw: pd.DataFrame):
    """Hojas con una columna de año y otra de mes (número o nombre). Devuelve (fechas, {cols}) o None."""
    def etiqueta(c):
        return _norm(" ".join(str(x) for x in raw.iloc[:15, c] if isinstance(x, str)))
    col_anio = col_mes = None
    meses = None
    for c in raw.columns:
        num = pd.to_numeric(raw[c], errors="coerce").astype("float64")
        e = etiqueta(c)
        if col_anio is None and re.search(r"\b(ano|anio|year)\b", e) and num.between(2000, 2100).sum() >= 24:
            col_anio, anios = c, num.where(num.between(2000, 2100))
        elif col_mes is None and re.search(r"\b(mes|month)\b", e):
            texto = raw[c].astype("string").str.strip().str.lower().str[:3].map(_MESES)
            m_txt = pd.to_numeric(texto, errors="coerce").astype("float64")
            m_num = num.where(num.between(1, 12))
            m = m_num if m_num.notna().sum() >= m_txt.notna().sum() else m_txt
            if m.notna().sum() >= 24:
                col_mes, meses = c, m
    if col_anio is None or col_mes is None:
        return None
    f = pd.to_datetime(pd.DataFrame({"year": anios, "month": meses, "day": 1}), errors="coerce")
    return (f, {col_anio, col_mes}) if _mensual_valida(f) else None


def _vista_previa(hoja: str, raw: pd.DataFrame) -> str:
    return f"[hoja '{hoja}', {raw.shape[0]} filas x {raw.shape[1]} columnas]\n" + raw.iloc[:12, :8].to_string(max_colwidth=18)


def _buscar_archivos_locales(nombre: str, extensiones=(".xls", ".xlsx", ".xlsm")) -> tuple[list[Path], list[Path]]:
    """
    Busca Excel cuyo nombre empiece con `nombre` (sin distinguir mayúsculas: ICG.xlsx, icg.xls, ICG_2026.xlsx)
    en DIR_EXCEL_EXTRA (si se configuró), raw/, data/raw/, data/ y la carpeta actual, y lo mismo
    relativo a la carpeta donde está este script. Ignora temporales de Excel (~$...) y descargas automáticas.
    """
    bases = [Path(".")]
    try:
        bases.append(Path(__file__).resolve().parent)
    except NameError:                       # ejecutado con exec()/pegado en una celda: no hay __file__
        pass
    candidatas: list[Path] = [Path(DIR_EXCEL_EXTRA)] if DIR_EXCEL_EXTRA else []
    candidatas += [DIR_RAW, DIR_OUT]
    for b in bases:
        candidatas += [b / "raw", b / "data" / "raw", b / "data", b]
    carpetas: list[Path] = []
    for d in candidatas:
        if d.is_dir() and all(d.resolve() != c.resolve() for c in carpetas):
            carpetas.append(d)
    hallados = [p for d in carpetas for p in d.iterdir()
                if p.is_file() and not p.name.startswith(("~$", ".")) and "_descargado" not in p.name.lower()
                and p.name.lower().startswith(nombre) and p.suffix.lower() in extensiones]
    return hallados, carpetas


def _obtener_archivo_utdt(nombre: str) -> Path:
    """Devuelve el Excel cargado a mano (ICC.xlsx / ICG.xlsx). No descarga nada de internet."""
    hallados, carpetas = _buscar_archivos_locales(nombre)
    if hallados:
        # preferencia: nombre exacto configurado (ICG.xlsx / ICC.xlsx) > .xlsx > el más reciente
        elegido = max(hallados, key=lambda p: (p.name in ARCHIVOS_LOCALES.get(nombre, ()),
                                               p.suffix.lower() == ".xlsx", p.stat().st_mtime))
        log.info("%s: usando archivo local %s", nombre.upper(), elegido)
        otros = [p.name for p in hallados if p != elegido]
        if otros:
            log.warning("%s: había otros archivos candidatos (%s). Si el elegido no es el correcto, "
                        "borrá los demás o renombralo.", nombre.upper(), ", ".join(otros))
        return elegido
    raise FileNotFoundError(
        f"Falta el archivo de {nombre.upper()}. Copialo con un nombre que empiece con '{nombre.upper()}' "
        f"(ej. {ARCHIVOS_LOCALES[nombre][0]}) a alguna de estas carpetas: {[str(c.resolve()) for c in carpetas]}")


def leer_indice_utdt(nombre: str) -> pd.DataFrame:
    """Lee el Excel y detecta columna de fecha y de valor. Valida el rango para no cargar la columna equivocada."""
    ruta = _obtener_archivo_utdt(nombre)
    lo, hi = UTDT[nombre]["rango"]
    try:
        xls = pd.ExcelFile(ruta)
    except ImportError as e:
        raise RuntimeError("Falta una librería para leer el Excel: ejecutá  !pip install xlrd openpyxl") from e
    previews = []
    for hoja in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=hoja, header=None)
        previews.append(_vista_previa(hoja, raw))
        # candidatos de fecha: una columna con fechas, o dos columnas (año y mes)
        candidatos = []
        for c in raw.columns:
            f = _leer_fechas(raw[c])
            if f is not None:
                candidatos.append((f, {c}))
        separados = _fechas_anio_mes_separados(raw)
        if separados is not None:
            candidatos.append(separados)
        if not candidatos:
            continue
        fechas, cols_fecha = max(candidatos, key=lambda t: t[0].notna().sum())
        valid = fechas.notna() & (fechas >= "2000-01-01") & (fechas <= "2100-01-01")
        cands = []   # columna de valor: preferir la que mencione el nombre en el encabezado; si no, la primera numérica
        for c in raw.columns:
            if c in cols_fecha:
                continue
            vals = pd.to_numeric(raw.loc[valid, c], errors="coerce")
            if vals.notna().mean() < 0.8:
                continue
            etiqueta = _norm(" ".join(str(x) for x in raw.iloc[:15, c] if isinstance(x, str)))
            cands.append((nombre in etiqueta, c, vals))
        if not cands:
            continue
        cands.sort(key=lambda t: not t[0])
        _, col_val, vals = cands[0]
        df = pd.DataFrame({"periodo": fechas[valid].dt.to_period("M").values, nombre: vals.values}).dropna()
        if not df[nombre].between(lo, hi).all():
            raise ValueError(f"{nombre.upper()}: valores fuera de [{lo}, {hi}] en la hoja '{hoja}', columna {col_val}. "
                             "Probablemente se eligió la columna equivocada; revisá el Excel.")
        log.info("%s: hoja '%s', fecha=col %s, valor=col %s, %d meses (%s → %s)", nombre.upper(), hoja,
                 sorted(cols_fecha), col_val, len(df), df["periodo"].min(), df["periodo"].max())
        return df.drop_duplicates("periodo", keep="last")
    raise ValueError(f"No pude detectar fechas mensuales y valores en {ruta}. Así se ve el archivo "
                     f"(pasame esto o el archivo para ajustar la lectura):\n" + "\n".join(previews))


# ───────────────────────────── CONSOLIDACIÓN ─────────────────────────────
def consolidar(base: pd.DataFrame, series: list[pd.DataFrame]) -> pd.DataFrame:
    """base: DataFrame con 'fecha'. series: DataFrames (disponible_desde, <variable>)."""
    out = base.copy()
    out["fecha"] = _ns(out["fecha"])
    out = out.sort_values("fecha").reset_index(drop=True)
    for s in series:
        s = s.drop_duplicates("disponible_desde", keep="last").sort_values("disponible_desde")
        out = pd.merge_asof(out, s, left_on="fecha", right_on="disponible_desde", direction="backward")
        out = out.drop(columns="disponible_desde")
    return out


def agregar_derivadas(df: pd.DataFrame, ultima_merval=None, ultima_argt=None) -> pd.DataFrame:
    df = df.sort_values("fecha").reset_index(drop=True).copy()
    # Variación diaria (%) del Merval empalmado. Sábados, domingos y feriados valen 0 (el nivel no cambió) y el
    # primer día hábil siguiente trae la variación acumulada desde el último cierre. Después del ÚLTIMO cierre
    # disponible (p. ej. falta cargar raw/merval_msci.csv) queda vacía, en vez de repetir el último nivel como "0 %".
    if "merval_nivel" in df.columns:
        nivel = df["merval_nivel"]
        df["merval_var_pct"] = (nivel / nivel.shift(1) - 1) * 100
        if ultima_merval is not None:
            df.loc[df["fecha"] > pd.Timestamp(ultima_merval), "merval_var_pct"] = np.nan
    if "argt_nivel" in df.columns:               # igual criterio para el ETF ARGT (mercado de EE.UU.: sin datos en sus feriados → 0 %)
        nivel = df["argt_nivel"]
        df["argt_var_pct"] = (nivel / nivel.shift(1) - 1) * 100
        if ultima_argt is not None:
            df.loc[df["fecha"] > pd.Timestamp(ultima_argt), "argt_var_pct"] = np.nan
    for c in COLUMNAS_BASE:            # si una fuente falló, la derivada queda en NaN en vez de romper el script
        if c not in df.columns:
            df[c] = np.nan
    df["brecha_pct"] = (df["blue_venta"] / df["dolar_oficial"] - 1) * 100

    # Rendimiento mensual del plazo fijo (30 días) a partir de la TNA
    rend_pf_mensual = df["tasa_plazo_fijo_minorista_tna"] * 30 / 365   # en %
    df["tasa_real_mensual"] = ((1 + rend_pf_mensual / 100) / (1 + df["inflacion_mensual_pct"] / 100) - 1) * 100
    return df


def auditar_dataset(df: pd.DataFrame) -> list[str]:
    """Controles de calidad sobre la tabla final. Devuelve la lista de problemas encontrados."""
    problemas = []
    if df["fecha"].duplicated().any():
        problemas.append("hay fechas duplicadas")
    if not df["fecha"].is_monotonic_increasing:
        problemas.append("las fechas no están ordenadas")
    if (df["blue_venta"] <= 0).any():
        problemas.append("blue_venta tiene valores <= 0")
    vacias = [c for c in df.columns if df[c].isna().all()]
    if vacias:
        problemas.append(f"columnas totalmente vacías: {vacias}")
    resumen = []
    for c in df.columns.drop("fecha"):
        validos = df[c].dropna()
        if validos.empty:
            continue
        primera = df.loc[validos.index[0], "fecha"].date()
        faltan_post = df.loc[validos.index[0]:, c].isna().mean() * 100
        if faltan_post > 30:
            problemas.append(f"{c}: {faltan_post:.0f}% de faltantes aun después de su primer dato")
        resumen.append(f"  {c:<32} desde {primera} | {df[c].isna().mean()*100:5.1f}% faltante | último = {validos.iloc[-1]:,.2f}")
    log.info("Resumen por columna:\n%s", "\n".join(resumen))
    return problemas


AYUDA_FORMATO_CSV = f"""FORMATO DE LOS ARCHIVOS CSV DE ESTE PROYECTO
Separador de columnas: "{CSV_SEP}"   |   Separador decimal: "{CSV_DECIMAL}" (coma)   |   Fechas: DD/MM/AAAA (día/mes/año)   |   Codificación: UTF-8 con BOM

pandas (Python):
    import pandas as pd
    df = pd.read_csv("dataset_dolar_blue.csv", sep="{CSV_SEP}", decimal="{CSV_DECIMAL}", encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], format="{CSV_FECHA}")
    # o, con el script:  import construir_dataset_dolar_blue as m;  df = m.leer_dataset()

R:
    df <- read.csv2("dataset_dolar_blue.csv", fileEncoding = "UTF-8-sig")
    df$fecha <- as.Date(df$fecha, format = "%d/%m/%Y")

Excel:
    Datos > Obtener datos > Desde un archivo de texto/CSV > Delimitador: Punto y coma.
    Las fechas DD/MM/AAAA se reconocen como fecha si Excel está configurado en español (Argentina).
    (Con Windows/Excel en español suele abrirse directo con doble clic.)

Power BI:
    Obtener datos > Texto/CSV > Delimitador: Punto y coma, Origen del archivo: 65001 UTF-8.
    Si los decimales salen mal: Transformar datos > Tipo de datos > Usar configuración regional > Español (Argentina)
    (también sirve para que la columna fecha se lea como día/mes/año).

Si leés el archivo con "," como separador o "." como decimal, las columnas salen pegadas o los números como texto.
"""


def _escribir_leeme(carpeta: Path) -> None:
    (carpeta / "LEEME_formato_dataset.txt").write_text(AYUDA_FORMATO_CSV, encoding="utf-8")


def leer_dataset(ruta: str | Path | None = None) -> pd.DataFrame:
    """Lee dataset_dolar_blue.csv con el formato correcto (sep=";", decimal=","); fecha como datetime. Sin ruta, lo busca."""
    ruta = Path(ruta) if ruta else _buscar_dataset_csv()
    df = _leer_csv(ruta)
    df["fecha"] = _ns(df["fecha"])
    return df


def _guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    df = df.copy()
    for c in df.columns:                                   # toda columna de fechas se escribe como día/mes/año
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime(CSV_FECHA)
    df.to_csv(ruta, index=False, sep=CSV_SEP, decimal=CSV_DECIMAL, encoding="utf-8-sig")
    _escribir_leeme(ruta.parent)


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee el CSV detectando su formato: ';' + coma decimal (el actual) o ',' + punto decimal (versiones anteriores)."""
    with open(ruta, encoding="utf-8-sig") as f:
        encabezado = f.readline()
    sep, dec = (";", ",") if ";" in encabezado else (",", ".")
    df = pd.read_csv(ruta, sep=sep, decimal=dec, encoding="utf-8-sig")
    if "fecha" in df.columns:                              # DD/MM/AAAA (actual) o AAAA-MM-DD (versiones anteriores); sin adivinar
        primera = str(df["fecha"].dropna().iloc[0]) if df["fecha"].notna().any() else ""
        df["fecha"] = pd.to_datetime(df["fecha"], format=CSV_FECHA if "/" in primera else "%Y-%m-%d")
    return df


def _buscar_dataset_csv() -> Path:
    bases = [Path(".")]
    try:
        bases.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidatas = [DIR_OUT / "dataset_dolar_blue.csv"]
    for b in bases:
        candidatas += [b / "data" / "dataset_dolar_blue.csv", b / "dataset_dolar_blue.csv"]
    for c in candidatas:
        if c.is_file():
            return c
    raise FileNotFoundError("No encontré dataset_dolar_blue.csv. Busqué en: "
                            + ", ".join(str(c.resolve()) for c in candidatas)
                            + ". Pasá la ruta: actualizar_icc_icg('/ruta/dataset_dolar_blue.csv')")


def _log_arrastre(fechas: pd.Series, ultimos: dict) -> None:
    """Informa, para ICG e ICC, cuántas filas usan el último valor disponible por no existir un dato más nuevo."""
    for n, ultimo in ultimos.items():
        meses, dia = PUBLICACION_MENSUAL[n]
        # primer día en que correspondería el valor del mes siguiente al último que trae el Excel
        sig = (ultimo + 1).to_timestamp() + pd.DateOffset(months=meses) + pd.Timedelta(days=dia - 1)
        n_arr = int((fechas >= sig).sum())
        if n_arr:
            log.info("%s: último mes en el Excel = %s. %d filas (desde %s) usan ese último valor disponible, "
                     "porque todavía no hay dato más nuevo.", n.upper(), ultimo, n_arr, sig.date())
        else:
            log.info("%s: último mes en el Excel = %s; sin filas arrastradas.", n.upper(), ultimo)


def actualizar_icc_icg(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """
    Lee ICC.xlsx e ICG.xlsx (cargados a mano en raw/), los une por fecha al dataset y REEMPLAZA las
    columnas icc e icg en dataset_dolar_blue.csv. No toca ninguna otra columna ni consulta internet.

    Unión por fecha: cada día toma el valor de su mismo mes en el Excel. Si el Excel no tiene ese mes todavía
    (los meses posteriores al último dato), usa el ÚLTIMO valor disponible; nunca inventa valores.
    Si preferís que cada valor rija recién desde el mes siguiente (evita usar un dato antes de su publicación),
    poné (1, 1) en PUBLICACION_MENSUAL["icg"] y ["icc"].
    """
    ruta = Path(ruta_csv) if ruta_csv else _buscar_dataset_csv()
    df = _leer_csv(ruta)
    df = df.drop(columns=[c for c in COLUMNAS_ELIMINADAS if c in df.columns])
    df["fecha"] = _ns(df["fecha"])
    antes = {c: df[c].copy() if c in df.columns else pd.Series(np.nan, index=df.index) for c in ("icg", "icc")}
    log.info("Dataset a actualizar: %s (%d filas, %s → %s)", ruta, len(df), df["fecha"].min().date(), df["fecha"].max().date())

    idx = {n: leer_indice_utdt(n) for n in ("icg", "icc")}                    # si falta un Excel, falla con mensaje claro
    series = [_mensual(idx[n], n) for n in idx]
    nuevo = consolidar(df.drop(columns=[c for c in ("icg", "icc") if c in df.columns]), series)

    for c in ORDEN_COLUMNAS:
        if c not in nuevo.columns:
            nuevo[c] = np.nan
    extras = [c for c in nuevo.columns if c not in ORDEN_COLUMNAS]
    nuevo = nuevo[ORDEN_COLUMNAS + extras]

    copia = ruta.with_name(ruta.stem + "_backup" + ruta.suffix)           # copia de seguridad de lo que había
    shutil.copyfile(ruta, copia)
    _guardar_csv(nuevo, ruta)

    _log_arrastre(nuevo["fecha"], {n: idx[n]["periodo"].max() for n in idx})
    for c in ("icg", "icc"):
        v = nuevo[c]
        cambiaron = int((~np.isclose(v.fillna(-999), antes[c].reset_index(drop=True).fillna(-999).values)).sum())
        con_dato = v.dropna()
        log.info("%s: %d de %d filas con valor | primera fecha con dato %s | último valor %.2f | filas modificadas %d",
                 c.upper(), len(con_dato), len(v), nuevo.loc[con_dato.index[0], "fecha"].date(), con_dato.iloc[-1], cambiaron)
    log.info("Guardado en %s (copia anterior: %s)", ruta, copia.name)
    return nuevo


def main() -> pd.DataFrame:
    log.info("Script versión %s | BCRA_BASE = %s", VERSION, BCRA_BASE)
    if "/v4.0/" not in BCRA_BASE:
        log.warning("BCRA_BASE no apunta a la v4.0 (según el manual, v1.0 y v2.0 están dadas de baja y devuelven 404).")
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    errores: dict[str, str] = {}
    series: list[pd.DataFrame] = []
    ultima_merval = ultima_argt = None

    def intentar(nombre, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            errores[nombre] = str(e)
            log.error("%s: %s", nombre, e)
            return None

    # Base: el dólar blue (variable objetivo) define el calendario del dataset
    blue = obtener_dolar("blue", "blue_venta", FUENTE_BLUE)
    base = blue[blue["fecha"] >= FECHA_INICIO].drop_duplicates("fecha", keep="last")

    # Diarias de mercado
    for nombre, fn in [
        ("dolar_oficial", lambda: obtener_dolar("oficial", "dolar_oficial", FUENTE_OFICIAL)),
        ("dolar_mep", lambda: obtener_dolar("mep", "dolar_mep", FUENTE_MEP_CCL)),
        ("dolar_ccl", lambda: obtener_dolar("ccl", "dolar_ccl", FUENTE_MEP_CCL)),
        ("riesgo_pais_pb", lambda: serie_fecha_valor("/finanzas/indices/riesgo-pais", "riesgo_pais_pb")),
        ("merval_nivel", merval),
        ("argt_nivel", argt),
    ]:
        df = intentar(nombre, fn)
        if df is not None:
            series.append(_diaria(df, nombre, REZAGO_DIARIO["mercado"]))
            if nombre == "merval_nivel":
                ultima_merval = df["fecha"].max()
            elif nombre == "argt_nivel":
                ultima_argt = df["fecha"].max()
            elif nombre in ("dolar_mep", "dolar_ccl"):
                primera = df.dropna(subset=[nombre])["fecha"].min()
                log.info("%s: datos desde %s (antes queda vacío; el dataset empieza el %s)", nombre,
                         primera.date(), pd.Timestamp(FECHA_INICIO).date())

    # Diarias del BCRA (v4.0)
    try:
        catalogo = bcra_listar_variables()
        cat_csv = catalogo.copy()
        for c in ("primerFechaInformada", "ultFechaInformada"):
            if c in cat_csv.columns:
                cat_csv[c] = pd.to_datetime(cat_csv[c], errors="coerce")
        _guardar_csv(cat_csv, DIR_OUT / "bcra_catalogo_v4.csv")
        ids = bcra_resolver_ids(catalogo)
        usadas = catalogo[catalogo["idVariable"].isin(ids.values())][["idVariable", "descripcion", "unidadExpresion", "periodicidad"]]
        _guardar_csv(usadas.assign(columna=usadas["idVariable"].map({v: k for k, v in ids.items()})),
                     DIR_OUT / "bcra_variables_usadas.csv")
        for col, id_var in ids.items():
            df = intentar(col, lambda i=id_var, c=col: bcra_serie(i, c))
            if df is not None:
                series.append(_diaria(df, col, REZAGO_DIARIO["bcra"]))
    except Exception as e:  # noqa: BLE001
        errores["bcra"] = str(e)
        log.error("BCRA: %s", e)

    # Mensuales
    infl = intentar("inflacion_mensual_pct", lambda: serie_fecha_valor("/finanzas/indices/inflacion", "inflacion_mensual_pct"))
    if infl is not None:
        infl["periodo"] = infl["fecha"].dt.to_period("M")
        series.append(_mensual(infl, "inflacion_mensual_pct"))
    ultimos_idx = {}
    for nombre in ("icg", "icc"):
        df = intentar(nombre, lambda n=nombre: leer_indice_utdt(n))
        if df is not None:
            series.append(_mensual(df, nombre))
            ultimos_idx[nombre] = df["periodo"].max()

    final = agregar_derivadas(consolidar(base, series), ultima_merval, ultima_argt)
    for c in ORDEN_COLUMNAS:
        if c not in final.columns:
            final[c] = np.nan
    final = final[ORDEN_COLUMNAS]

    ruta = DIR_OUT / "dataset_dolar_blue.csv"
    _guardar_csv(final, ruta)
    _log_arrastre(final["fecha"], ultimos_idx)
    log.info("Dataset: %d filas, %s → %s | guardado en %s", len(final),
             final["fecha"].min().date(), final["fecha"].max().date(), ruta)

    problemas = auditar_dataset(final)
    for col in ("dolar_mep", "dolar_ccl"):                  # obligatorias: el modelo único las usa siempre
        if final[col].isna().all():
            log.error("%s quedó vacía: el modelo usa MEP y CCL y no podrá entrenarse. Revisá FUENTE_MEP_CCL y la conexión.", col)
            problemas.append(f"{col} vacía (obligatoria para el modelo)")
    if errores:
        log.warning("Fuentes con error (columnas vacías): %s", list(errores))
    if problemas:
        log.warning("Auditoría del dataset, puntos a revisar:\n  - %s", "\n  - ".join(problemas))
    else:
        log.info("Auditoría del dataset: sin problemas detectados")
    return final


if __name__ == "__main__":
    main()
