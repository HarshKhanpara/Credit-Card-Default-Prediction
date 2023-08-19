from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))
    
def intTypecaster(x):
    return int(x)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    PAY_0 = request.json['PAY_0']
    AGE = request.json['AGE']
    BILL_AMT1 = request.json['BILL_AMT1']
    BILL_AMT2 = request.json['BILL_AMT2']
    BILL_AMT3 = request.json['BILL_AMT3']
    BILL_AMT4 = request.json['BILL_AMT4']
    BILL_AMT5 = request.json['BILL_AMT5']
    BILL_AMT6 = request.json['BILL_AMT6']
    LIMIT_BAL = request.json['LIMIT_BAL']
    EDUCATION = request.json['EDUCATION']
    MARRIAGE = request.json['MARRIAGE']
    PAY_2 = request.json['PAY_2']
    PAY_3 = request.json['PAY_3']
    PAY_4 = request.json['PAY_4']
    PAY_5 = request.json['PAY_5']
    PAY_6 = request.json['PAY_6']
    PAY_AMT1 = request.json['PAY_AMT1']
    PAY_AMT2 = request.json['PAY_AMT2']
    PAY_AMT3 = request.json['PAY_AMT3']
    PAY_AMT4 = request.json['PAY_AMT4']
    PAY_AMT5 = request.json['PAY_AMT5']
    PAY_AMT6 = request.json['PAY_AMT6']
    ID = request.json['ID']
    SEX = request.json['SEX']
    
    # Sum of all the pay amounts
    sum_pay_amt = intTypecaster(PAY_AMT1) + intTypecaster(PAY_AMT2) + intTypecaster(PAY_AMT3) + intTypecaster(PAY_AMT4) + intTypecaster(PAY_AMT5) + intTypecaster(PAY_AMT6)
    
    # Sum of all the bill amounts
    sum_bill_amt = intTypecaster(BILL_AMT1) + intTypecaster(BILL_AMT2) + intTypecaster(BILL_AMT3) + intTypecaster(BILL_AMT4) + intTypecaster(BILL_AMT5) + intTypecaster(BILL_AMT6)
    
    # Division of sum of pay amounts and sum of bill amounts
    division = sum_pay_amt/sum_bill_amt
    
    
    # Make prediction using model loaded from disk as per the data.
    predictions = model.predict([[ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,sum_pay_amt,sum_bill_amt,division]])
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)