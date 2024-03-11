from flask import Flask, request, jsonify
from score import *
import pickle
app = Flask(__name__)
# Load the trained model and vectorizer
loaded_model = pickle.load(open("D:\Applied ML\ASSINGMENT_3\Best_models\logistic_regression_best_model.pkl", 'rb'))
@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text')
    if text:
        prediction, propensity = score(text, loaded_model, threshold=0.5)
        return jsonify({'prediction': prediction, 'propensity': propensity})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True,port=5000)