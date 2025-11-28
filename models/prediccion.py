from pathlib import Path
import joblib
import pandas as pd

# ===============================
# Cargar modelo y configuración
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "car_model_pack.pkl"

_pack = joblib.load(PACK_PATH)

modelo = _pack["modelo"]
FEATURES = _pack["feature_names"]

# Mapeo inverso de clases
map_clase_inv = {0: "unacc", 1: "acc", 2: "good", 3: "vgood"}

# ===============================
# Función principal de predicción
# ===============================

def predecir_auto(data: dict):
    """
    Recibe un dict con las variables del vehículo.
    Devuelve un dict con la predicción en número y texto.
    """

    # ------------------------
    # Validar formato JSON
    # ------------------------
    if not isinstance(data, dict):
        return {"ok": False, "msg": "El JSON debe ser un dict válido."}

    # Convertir a DataFrame
    df = pd.DataFrame([data])

    # ------------------------
    # Validar columnas faltantes
    # ------------------------
    faltan = [c for c in FEATURES if c not in df.columns]
    if faltan:
        return {
            "ok": False,
            "msg": "Faltan columnas para predecir.",
            "faltan": faltan,
            "esperadas": FEATURES
        }

    # Reordenar columnas como en entrenamiento
    X = df.reindex(columns=FEATURES).copy()

    # ------------------------
    # Convertir valores a numérico
    # ------------------------
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Validar NaN
    if X.isna().any().any():
        cols_nan = X.columns[X.isna().any()].tolist()
        return {
            "ok": False,
            "msg": "Hay valores no numéricos o inválidos.",
            "columnas_con_problema": cols_nan
        }

    # ------------------------
    # Hacer la predicción
    # ------------------------
    try:
        pred = int(modelo.predict(X)[0])
        pred_texto = map_clase_inv.get(pred, "desconocido")
        return {
            "ok": True,
            "prediccion_num": pred,
            "PREDICCION_ESPERADA": pred_texto
        }
    except Exception as e:
        return {"ok": False, "msg": f"Error al predecir: {e}"}
