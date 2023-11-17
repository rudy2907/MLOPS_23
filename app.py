from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the input data is in JSON format
        input_data = request.get_json()

        # Perform your model prediction or processing here
        # For demonstration, echo the input data
        result = {"input_data": input_data}

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the Flask app on host 0.0.0.0 to make it externally accessible
    app.run(host='0.0.0.0', port=80)
