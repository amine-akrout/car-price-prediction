# Import Libraries
from __future__ import print_function
import sys
import mlflow
from flask import Flask, render_template, request
import pandas as pd


logged_model = './model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


app = Flask(__name__)

@app.route('/')
def entry_page():
    # Nicepage template of the webpage
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def render_message():
    try:
        # Get data input
        CarBrand = request.form['CarBrand']
        fueltype = request.form['fueltype']
        aspiration = request.form['aspiration']
        doornumber = request.form['doornumber']
        carbody = request.form['carbody']
        drivewheel = request.form['drivewheel']
        enginelocation = request.form['enginelocation']
        wheelbase = float(request.form['wheelbase'])
        carlength = float(request.form['carlength'])
        carwidth = float(request.form['carwidth'])
        carheight = float(request.form['carheight'])
        curbweight = int(request.form['curbweight'])
        enginetype = request.form['enginetype']
        cylindernumber = request.form['cylindernumber']
        enginesize = int(request.form['enginesize'])
        fuelsystem = request.form['fuelsystem']
        boreratio = float(request.form['boreratio'])
        horsepower = int(request.form['horsepower'])
        citympg = int(request.form['citympg'])
        highwaympg = int(request.form['highwaympg'])
        data = [[CarBrand, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation,  wheelbase, carlength,
        carwidth, carheight, curbweight, enginetype, cylindernumber, enginesize, fuelsystem,  boreratio, horsepower,
        citympg, highwaympg]]
        df = pd.DataFrame(data, columns = ['CarBrand', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                            'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight','curbweight', 'enginetype',
                            'cylindernumber', 'enginesize','fuelsystem', 'boreratio', 'horsepower', 'citympg', 'highwaympg'])

        preds = loaded_model.predict(pd.DataFrame(df))

        print('Python module executed successfully')
        message = 'Estimated price : {} '.format(round(preds[0],2))
        print(message, file=sys.stderr)

    except Exception as e:
        # Store error to pass to the web page
        message = "Error encountered. Try with other values. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(
            e.__class__, e.args, e.__doc__)

    # Return the model results to the web page
    return render_template('index.html' ,message=message)

if __name__ == '__main__':
    app.run(debug=True , host='localhost', port=8080)
