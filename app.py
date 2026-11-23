"""
Aplicación web del proyecto: pronóstico del dólar blue (Streamlit).  -- versión 2.1

Dos páginas:
  - vistas/inicio.py : el último valor conocido del blue y la proyección a dos semanas, con el aviso de alta incertidumbre (detector Isolation Forest)
                       y el gráfico de evolución. Pensada para cualquier persona.
  - vistas/modelo.py : cómo funciona el modelo, qué tan preciso fue y qué variables pesan (se llega desde el pie de la portada).
Las funciones compartidas (lectura de datos, formato y estilos) están en comun.py.
No entrena ni descarga nada: lee los CSV de data/, que se actualizan solos cada día hábil (GitHub Actions).

Ejecutar en una computadora:   streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Asegura que las páginas de vistas/ encuentren comun.py (está en la misma carpeta que este archivo)
RAIZ = str(Path(__file__).resolve().parent)
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

st.set_page_config(page_title="Pronóstico del dólar blue", page_icon="💵", layout="wide", initial_sidebar_state="collapsed")

pagina = st.navigation(
    [st.Page("vistas/inicio.py", title="Dólar blue: hoy y en dos semanas", icon="💵", default=True),
     st.Page("vistas/modelo.py", title="Cómo funciona el pronóstico", icon="📘", url_path="como-funciona")],
    position="hidden",
)
pagina.run()
