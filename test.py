import requests
import json

# URL base del servicio (ajusta según tu configuración)
BASE_URL = "http://localhost:5000"

def test_service():
    """Función para probar todos los endpoints del servicio"""
    
    print("🧪 Probando el servicio de clasificación de Iris\n")
    
    # 1. Probar endpoint principal
    print("1. Probando endpoint principal (/)")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 2. Probar health check
    print("2. Probando health check (/health)")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 3. Probar información del modelo
    print("3. Probando información del modelo (/model-info)")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 4. Probar predicción individual
    print("4. Probando predicción individual (/predict)")
    test_samples = [
        {
            "name": "Iris Setosa",
            "features": [5.1, 3.5, 1.4, 0.2]
        },
        {
            "name": "Iris Versicolor", 
            "features": [6.4, 3.2, 4.5, 1.5]
        },
        {
            "name": "Iris Virginica",
            "features": [6.3, 3.3, 6.0, 2.5]
        }
    ]
    
    for sample in test_samples:
        print(f"Probando {sample['name']}:")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"features": sample["features"]},
                headers={"Content-Type": "application/json"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                predicted_class = result['prediction']['class']
                confidence = result['confidence']
                print(f"Predicción: {predicted_class} (Confianza: {confidence:.3f})")
            else:
                print(f"Error: {response.json()}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")
    
    # 5. Probar predicción batch
    print("5. Probando predicción batch (/predict-batch)")
    batch_data = {
        "samples": [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [6.4, 3.2, 4.5, 1.5],  # Versicolor
            [6.3, 3.3, 6.0, 2.5],  # Virginica
            [5.8, 2.7, 5.1, 1.9]   # Virginica
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict-batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Total de muestras procesadas: {result['total_samples']}")
            for res in result['results']:
                idx = res['sample_index']
                pred_class = res['prediction']['class']
                conf = res['confidence']
                print(f"  Muestra {idx}: {pred_class} (Confianza: {conf:.3f})")
        else:
            print(f"Error: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 6. Probar error handling
    print("6. Probando manejo de errores")
    
    # Datos incompletos
    print("6.1. Probando con datos incompletos:")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"features": [5.1, 3.5, 1.4]},  # Solo 3 características
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Sin datos
    print("6.2. Probando sin datos:")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")

def quick_test():
    """Prueba rápida con una sola predicción"""
    print("🚀 Prueba rápida del servicio\n")
    
    # Ejemplo de flor Iris Setosa
    sample = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción exitosa!")
            print(f"Características: {result['input_features']}")
            print(f"Predicción: {result['prediction']['class']}")
            print(f"Confianza: {result['confidence']:.3f}")
            print(f"Probabilidades: {result['probabilities']}")
        else:
            print(f"❌ Error: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar al servicio.")
        print("Asegúrate de que el servicio Flask esté ejecutándose.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        test_service()
        
    print("\n🎯 Ejemplos de uso con curl:")
    print("curl -X POST http://localhost:5000/predict \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"features\": [5.1, 3.5, 1.4, 0.2]}'")
    print()
    print("curl -X GET http://localhost:5000/model-info")