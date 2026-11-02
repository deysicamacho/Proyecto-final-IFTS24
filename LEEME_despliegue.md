# Aplicación web del dólar blue: cómo publicarla (Etapa 6)

## Qué hay en esta carpeta
| Archivo | Para qué sirve |
|---|---|
| `app.py` | La aplicación web (Streamlit). Solo lee los CSV de `data/`; no entrena ni descarga nada. |
| `actualizar_pipeline.py` | Ejecuta las 5 capas del proyecto en orden y deja en `data/` lo que muestra la app. |
| `construir_dataset_dolar_blue.py`, `diagnostico_dataset.py`, `analisis_eda_features.py`, `comparar_modelos.py`, `modelo_diario.py` | Los 5 scripts del pipeline, sin cambios. |
| `.github/workflows/actualizar_datos.yml` | Tarea programada de GitHub: corre el pipeline cada día hábil a las 19:07 y publica los resultados. |
| `requirements.txt` | Librerías de la app (las instala Streamlit Community Cloud). |
| `requirements-pipeline.txt` | Librerías del pipeline (las instala GitHub Actions). |
| `.streamlit/config.toml` | Colores de la app. |
| `raw/` | Acá van `ICC.xlsx` e `ICG.xlsx` (se cargan a mano). |
| `data/` | Resultados ya calculados con el dataset al 01/11/2026, para que la app funcione desde el primer día. |

## Cómo se conecta el pipeline con la app
```
GitHub Actions (cada día hábil, 19:07)
   └─ actualizar_pipeline.py: datos → diagnóstico → variables → (lunes: comparación) → predicción h=1 y h=10
        └─ si todo salió bien: guarda los CSV nuevos en el repositorio
             └─ Streamlit Community Cloud detecta el cambio y la app muestra los datos nuevos
```
Si algo falla (por ejemplo, el MEP o el CCL vienen vacíos), no se publica nada y la app sigue mostrando la última
actualización correcta. GitHub te avisa por correo.

## Paso 1: crear el repositorio en GitHub
1. Entrá a github.com (creá una cuenta si no tenés) y tocá **New repository**.
2. Nombre: por ejemplo `dolar-blue-app`. Elegí **Public** y tocá **Create repository**.

## Paso 2: subir los archivos
1. Descomprimí el zip en tu computadora.
2. En el repositorio, tocá **Add file → Upload files** y arrastrá **todo el contenido** de la carpeta (archivos y carpetas).
3. Tocá **Commit changes**.

**Ojo con las carpetas que empiezan con punto** (`.github` y `.streamlit`): en Mac y a veces en Windows están ocultas y no
se suben al arrastrar. Revisá que aparezcan en el repositorio. Si falta alguna, creala a mano: **Add file → Create new
file**, escribí como nombre `.github/workflows/actualizar_datos.yml`, pegá el contenido del archivo y guardá. Lo mismo con
`.streamlit/config.toml`.

## Paso 3: subir ICC e ICG
**Add file → Upload files** → arrastrá `ICC.xlsx` e `ICG.xlsx` dentro de la carpeta `raw/` → **Commit changes**.
(Para subirlos dentro de `raw/`, primero entrá a la carpeta `raw` en GitHub y después tocá Add file.)

## Paso 4: permitir que la tarea programada guarde resultados
**Settings → Actions → General → Workflow permissions** → elegí **Read and write permissions** → **Save**.

## Paso 5: probar la actualización automática
1. Pestaña **Actions** → **Actualizar datos del dólar blue** → **Run workflow** → **Run workflow**.
2. Esperá unos 10 minutos. Si termina en verde, vas a ver un commit nuevo "Actualización automática de datos …".
3. Si termina en rojo, abrí la corrida y mirá el paso "Ejecutar el pipeline": el mensaje final dice qué falló.

Es la primera vez que el pipeline corre desde servidores de GitHub (en Estados Unidos). Si alguna fuente no responde desde
ahí, el registro lo indica: si son el blue, el MEP o el CCL la corrida se detiene; si es otra fuente, sigue y la app muestra
un aviso.

## Paso 6: publicar la app en Streamlit Community Cloud (gratis)
1. Entrá a share.streamlit.io e ingresá con tu cuenta de GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository: `tu-usuario/dolar-blue-app` · Branch: `main` · Main file path: `app.py`.
4. En **App URL** podés elegir la dirección (por ejemplo `dolar-blue-ml.streamlit.app`).
5. En **Advanced settings**, elegí Python 3.12. Tocá **Deploy**.
6. En unos minutos la app queda pública en esa dirección. Cada vez que GitHub Actions guarda datos nuevos, la app se actualiza sola.

## Mantenimiento
- **Cada mes**: cuando la Universidad Di Tella publique el ICC y el ICG nuevos, reemplazá los Excel en `raw/` (Paso 3).
- **App dormida**: si nadie la visita durante 12 horas, Streamlit la pone a dormir. Cualquier visitante la despierta con el
  botón "Yes, get this app back up!" (tarda alrededor de un minuto).
- **Tarea programada**: en repositorios públicos, GitHub desactiva las tareas programadas tras 60 días sin actividad. Los
  commits diarios deberían evitarlo; si ves que dejó de correr, reactivala en **Actions**.

## Probar en tu computadora (opcional)
```
pip install -r requirements.txt
streamlit run app.py
```
Para regenerar los datos: `pip install -r requirements-pipeline.txt` y `python actualizar_pipeline.py`.
Sin internet, con el dataset que ya está en `data/`: `python actualizar_pipeline.py --sin-descarga`.
