from flask import Flask, request, jsonify, abort
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('trained_model.pkl', 'rb') as file:
     model = pickle.load(file)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Import column names (encoded)
    filename = 'troop_movements.csv'
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)
    df = df[['unit_type', 'homeworld']]
    df = pd.get_dummies(df)
    columns = df.columns
    df = pd.DataFrame(columns=columns)

    try:
        # Get request data
        data = request.get_json(force=True)

         # Ensure the data is a list
        if isinstance(data, dict):
            data = [data]

        new_df = pd.DataFrame(data)
        new_df = pd.get_dummies(new_df)
        df = pd.concat([df, new_df], ignore_index=False)
        df = df.fillna(False)
        print(df)

        # Make a prediction
        prediction = model.predict(df)

        # Return the prediction
        return jsonify(prediction.tolist())
    except Exception as e:
        print(f"An error occurred: {e}")
        abort(500)

@app.route('/')
def unauth():
    abort(404)

if __name__ == '__main__':
    app.run(port=1977)