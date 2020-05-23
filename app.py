from flask import Flask, render_template,request
import pickle
import numpy as np
from sklearn.externals import joblib

water_point = ['Yes – Functional (and in use)','Yes – Functional (but not in use)',
               'Yes - But damaged','No - Broken down']


app = Flask(__name__)
model = pickle.load(open('SVM_trained_model_20F_SL.p','rb'))

def model_predict(data):
    probas = model.predict_proba(data)[0]
    max = np.argmax(probas)
    return int(max)

# *****************************
# WEBHOOK MAIN ENDPOINT : START
# *****************************
@app.route('/', methods=['GET','POST'])
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]
    #print(final_feature)

    Latitude             = final_feature[0][0]
    Longitude            = final_feature[0][1]
    Pump_Age             = final_feature[0][2]
    broke_down_repair    = final_feature[0][3]
    owns_water_point     = final_feature[0][4]
    management_committee = final_feature[0][5]
    extraction_type      = final_feature[0][6]
    waterpoint_type      = final_feature[0][7]

    final_data = [Latitude,Longitude,broke_down_repair,owns_water_point,management_committee,Pump_Age,extraction_type,waterpoint_type]

    scaler_filename = "sierra_scaler.save"
    scaler          = joblib.load(scaler_filename) 
    data            = np.array(final_data).reshape(-1, 8)
    Data_scaled     = scaler.transform(data)
    #print('Data_scaled \n', Data_scaled[0])

    prediction = model_predict(Data_scaled)

    print('Prediction : ', prediction)
    
    output = water_point[prediction]

    return render_template('index.html',prediction_text='Waterpoint Fnctionality :  {}'.format(output))
    #return render_template('index.html',prediction_text='Data_scaled : {}'.format(Latitude))

# ***************************
# WEBHOOK MAIN ENDPOINT : END
# ***************************

if __name__ == '__main__':
   app.run(debug=True)

#app.run(debug=True, host='0.0.0.0', port=5000)