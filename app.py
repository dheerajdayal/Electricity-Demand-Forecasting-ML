


from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)


modelpipeline = pickle.load(open(
    r'C:\Users\Lenovo\Desktop\Electircity_consumbtionp_prediction\model\Electricity_consumption_prediction.ppkl',
    'rb'
))


@app.route("/home")
def home():
    return render_template("index.html")  


@app.route("/predict", methods=['POST'])
def predict():
      
        features = [
            float(request.form['Global_reactive_power']),
            float(request.form['Voltage']),
            float(request.form['Sub_metering_1']),
            float(request.form['Sub_metering_2']),
            float(request.form['Sub_metering_3']),
            int(request.form['Hour']),
            int(request.form['DayOfWeek']),
            int(request.form['Month']),
            int(request.form['IsWeekend']),
            float(request.form['Rolling_3']),
            float(request.form['Rolling_5'])
        ]
        
       
        input_data = np.array([features])
        
        prediction = modelpipeline.predict(input_data)
        
        output = round(prediction[0], 2)
        return render_template("index.html", prediction_text=f" Predicted Power: {output} kW")
    



if __name__ == "__main__":

    app.run(debug=True)

