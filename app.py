from flask import Flask, request, jsonify
from pipeline.model_predictor import ModelPredictor
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.env = "development"
print("I am in flask app")



@app.route('/', methods=['GET'])
def index():
    return "Send a POST request to /predict with JSON payload to get a prediction."

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400

            try:
                # Initialize the model predictor
                model_predictor = ModelPredictor()
                prediction = model_predictor.get_prediction(data)
                return jsonify({"prediction": prediction})
                
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid request method"}), 405


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
