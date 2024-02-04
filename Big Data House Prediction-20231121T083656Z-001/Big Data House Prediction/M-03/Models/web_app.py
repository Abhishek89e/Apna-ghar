from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import joblib

app=Flask(__name__)

model_path = 'C:/Users/gts/House price predition using ML & NN/M3/random.pkl'

model = joblib.load(
    open(model_path,'rb'))

#model = pickle.load(
    #open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    
    bedrooms = (request.form['bedrooms'])
    bathrooms = (request.form['bathrooms'])
    sqft_living = (request.form['sqft_living'])
    sqft_lott = (request.form['sqft_lott'])
    floors = (request.form['floors'])
    #Asthma = float(request.form['Asthma'])
    waterfront = (request.form['waterfront'])
    view = (request.form['view'])
    condition = (request.form['condition'])
    grade = (request.form['grade'])
    sqft_above = (request.form['sqft_above'])
    sqft_basement = (request.form['sqft_basement'])
    yr_built = (request.form['yr_built'])
    yr_renovated=(request.form['yr_renovated'])
    lat =(request.form['lat'])
    sqft_living15 = (request.form['sqft_living15'])
    sqft_lot15 = (request.form['sqft_lot15'])
    month = (request.form['month'])
    year = (request.form['year'])
    
    query = np.array([[bedrooms, bathrooms, sqft_living, sqft_lott, floors, waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,lat,sqft_living15,sqft_lot15, month, year]])

    prediction = model.predict(query)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
