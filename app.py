from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
# Configurar CORS para permitir solicitudes desde el frontend
CORS(app, resources={r"/*": {"origins": "https://formulario-prueba-tzhl.onrender.com"}})
# Cargar el modelo y componentes al iniciar la aplicación
try:
    model = joblib.load('./iris_model.pkl')
    feature_selector = joblib.load('./feature_selector.pkl')
    class_names = joblib.load('./class_names.pkl')
    feature_names = joblib.load('./feature_names.pkl')
    print("Modelo cargado exitosamente")
except FileNotFoundError as e:
    print(f"Error: No se pudo cargar el modelo. Asegúrate de ejecutar primero el script de entrenamiento.")
    print(f"Archivo faltante: {e}")
    model = None

@app.route('/')
def home():
    """Página principal con información del servicio"""
    return {
        "message": "Servicio de Clasificación de Iris",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Realizar predicción",
            "/health": "GET - Estado del servicio",
            "/model-info": "GET - Información del modelo"
        }
    }

@app.route('/health')
def health():
    """Endpoint para verificar el estado del servicio"""
    if model is None:
        return jsonify({"status": "error", "message": "Modelo no cargado"}), 500
    return jsonify({"status": "ok", "message": "Servicio funcionando correctamente"})

@app.route('/model-info')
def model_info():
    """Información sobre el modelo"""
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    return jsonify({
        "model_type": "Logistic Regression",
        "features_used": feature_selector.n_features_,
        "total_features": len(feature_names),
        "feature_names": feature_names.tolist(),
        "selected_features": [feature_names[i] for i, selected in enumerate(feature_selector.support_) if selected],
        "classes": class_names.tolist()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Realizar predicción sobre nuevos datos"""
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No se proporcionaron datos"}), 400
        
        # Verificar si se proporcionaron las características
        if 'features' not in data:
            return jsonify({
                "error": "Se requiere el campo 'features'",
                "example": {
                    "features": [5.1, 3.5, 1.4, 0.2]
                },
                "feature_names": feature_names.tolist()
            }), 400
        
        features = data['features']
        
        # Verificar que se proporcionaron 4 características
        if len(features) != 4:
            return jsonify({
                "error": f"Se requieren exactamente 4 características, se proporcionaron {len(features)}",
                "feature_names": feature_names.tolist()
            }), 400
        
        # Convertir a numpy array y reshapear
        sample = np.array(features).reshape(1, -1)
        
        # Aplicar selección de características
        sample_selected = feature_selector.transform(sample)
        
        # Realizar predicción
        prediction = model.predict(sample_selected)[0]
        probabilities = model.predict_proba(sample_selected)[0]
        
        # Crear respuesta
        response = {
            "input_features": {
                name: value for name, value in zip(feature_names, features)
            },
            "prediction": {
                "class": class_names[prediction],
                "class_index": int(prediction)
            },
            "probabilities": {
                class_name: float(prob) for class_name, prob in zip(class_names, probabilities)
            },
            "confidence": float(max(probabilities))
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Realizar predicción sobre múltiples muestras"""
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({
                "error": "Se requiere el campo 'samples'",
                "example": {
                    "samples": [
                        [5.1, 3.5, 1.4, 0.2],
                        [6.2, 3.4, 5.4, 2.3]
                    ]
                }
            }), 400
        
        samples = np.array(data['samples'])
        
        if samples.shape[1] != 4:
            return jsonify({
                "error": f"Cada muestra debe tener exactamente 4 características",
                "feature_names": feature_names.tolist()
            }), 400
        
        # Aplicar selección de características
        samples_selected = feature_selector.transform(samples)
        
        # Realizar predicciones
        predictions = model.predict(samples_selected)
        probabilities = model.predict_proba(samples_selected)
        
        # Crear respuesta
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append({
                "sample_index": i,
                "input_features": samples[i].tolist(),
                "prediction": {
                    "class": class_names[pred],
                    "class_index": int(pred)
                },
                "probabilities": {
                    class_name: float(prob) for class_name, prob in zip(class_names, probs)
                },
                "confidence": float(max(probs))
            })
        
        return jsonify({
            "results": results,
            "total_samples": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": f"Error en la predicción batch: {str(e)}"}), 500

if __name__ == '__main__':
    # Verificar que los archivos del modelo existen
    required_files = ['iris_model.pkl', 'feature_selector.pkl', 'class_names.pkl', 'feature_names.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Archivos faltantes:")
        for file in missing_files:
            print(f"- {file}")
        print("\nEjecuta primero el script de entrenamiento para generar estos archivos.")
    else:
        print("Todos los archivos del modelo están presentes.")
        print("Iniciando servidor Flask...")
        print("Endpoints disponibles:")
        print("- GET /: Información general")
        print("- GET /health: Estado del servicio")
        print("- GET /model-info: Información del modelo")
        print("- POST /predict: Predicción individual")
        print("- POST /predict-batch: Predicción múltiple")
    
    app.run(debug=True, host='0.0.0.0', port=5000)