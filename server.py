from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_state_names', methods=['GET'])
def get_state_names():
    response = jsonify({
        'state': util.get_state_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_covid19', methods=['GET', 'POST'])
def predict_covid19():
    confirmed = int(request.form['confirmed'])
    state = request.form['state']
    year = int(request.form['year'])
    day = int(request.form['day'])

    response = jsonify({
        'covid_19_cured_cases': util.get_covid19_pred(state,confirmed,year,day)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For covid19 prediction...")
    util.load_saved_artifacts()
    app.run()