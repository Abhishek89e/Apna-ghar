from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import pandas as pd
import joblib
app=Flask(__name__)

model_path = 'C:/Users/gts/House price predition using ML & NN/M3/random.pkl'


model = joblib.load(
    open(model_path,'rb'))
print("model read")
@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    df=pd.read_excel('./test1.xlsx',header=None)
    #df=pd.read_excel('./test.xlsx',header=None)
    
   
    prediction=model.predict(df)

    #prediction=model.predict(query)
    #prediction = model.predict(query)
   
    return render_template('result.html',prediction=prediction)
   
if __name__=='__main__':
    app.run(debug=True)
    
                       
    
    














