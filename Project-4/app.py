from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the request
        input_data = request.json

        # Process the input data (convert gender to numeric, etc.)
        input_data['gender'] = int(input_data['gender'])
        # Add any other necessary processing for your input data

        # Make the prediction using the loaded model
        prediction = model.predict([[input_data['gender'], input_data['age'], input_data['bmi'],
                                     input_data['hba1c'], input_data['bloodGlucose']]])

        # Return the prediction as JSON
        return jsonify({'result': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
