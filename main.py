import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Path del model y el scaler
MODEL_PATH = 'model_test.pkl'
SCALER_PATH = 'scaler_test.pkl'

# Cargar el model y el scaler de los archivos
with open(MODEL_PATH, 'rb') as model_file, open(SCALER_PATH, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Chequear el modelo
print(model)

# Código para terminal
if len(sys.argv) > 1:
    arguments = sys.argv[1:]

    input_data = [float(arg) for arg in arguments]

    scaled_input = scaler.transform([input_data])

    # Predicción
    prediction = model.predict(scaled_input)

    print(f'Prediction: {prediction[0]}')
else:
    print('No arguments provided. Please provide input for prediction.')
