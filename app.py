from flask import Flask, request, jsonify
from models.prediccion import predecir_auto

app = Flask(__name__)

@app.route("/")
def home():
    return "API de predicción de autos funcionando ✅"

@app.route("/predecir", methods=["GET", "POST"])
def predecir():
    if request.method == "POST":
        # -------------------- POST --------------------
        data = request.get_json()
        if not data:
            return jsonify({"ok": False, "msg": "No se recibió JSON"}), 400
        
        resultado = predecir_auto(data)
        if resultado["ok"]:
            return jsonify(resultado)
        else:
            return jsonify(resultado), 400

    else:
        # -------------------- GET --------------------
        # Datos quemados
        datos_quemados = {
            "costo_compra": 4,
            "costo_mantenimiento": 2,
            "puertas": 4,
            "personas": 4,
            "tamano_baul": 2,
            "seguridad": 2
        }

        # Obtener la predicción usando la función del modelo
        prediccion = predecir_auto(datos_quemados)

        # Retornar solo los datos y la predicción simplificada
        resultado = {**datos_quemados, "PREDICCION_ESPERADA": prediccion["PREDICCION_ESPERADA"]}

        return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True)
