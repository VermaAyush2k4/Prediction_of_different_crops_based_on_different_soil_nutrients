from flask import Flask, render_template,request, jsonify
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder,StandardScaler

scaler = StandardScaler()

filename = 'xg_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__, static_folder="static")

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        ph = float(request.form['ph'])
        EC = float(request.form['EC'])
        S = float(request.form['S'])
        Cu = float(request.form['Cu'])
        Fe = float(request.form['Fe'])
        Mn = float(request.form['Mn'])
        Zn = float(request.form['Zn'])
        B = float(request.form['B'])
        modelSelected = int(request.form.get('selectModel'))
        
        def scale_input_row(input_row, scaler_path='scalar.pkl'):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            input_row = np.array(input_row)
            input_row = input_row.reshape(1, -1)
            scaled_row = scaler.transform(input_row)
            return scaled_row
        
        sample_input = [N,P,K,ph,EC,S,Cu,Fe,Mn,Zn,B]
        for val in sample_input:
          print(f"feature val = {val} and dtype = {type(val)}")
        filedata = ['dt_model.pkl' , 'et_model.pkl', 'mlp_model.pkl', 'rf_model.pkl' ,'xg_model.pkl' ]
        filename = filedata[modelSelected]
        model = pickle.load(open(filename, 'rb'))
        
        
        scaled_sample = scale_input_row(sample_input)
        sample_input = scaled_sample
        my_prediction = model.predict(sample_input)

    return render_template('result.html', prediction=my_prediction,selection = modelSelected)
        
        

if __name__ == '__main__':
	#app.run(debug=True)
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)