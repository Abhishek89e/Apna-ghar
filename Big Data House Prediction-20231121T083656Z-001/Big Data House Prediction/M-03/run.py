from flask import Flask,render_template,url_for,request
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

app=Flask(__name__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#get_ipython().run_line_magic('matplotlib', 'inline')

#sns.set()


# In[33]:


import warnings
warnings.simplefilter("ignore")


# In[34]:


data=pd.read_csv(r'kc_house_data.csv')


# In[35]:


# preview the data


# In[ ]:





# In[36]:


price_corr = data.corr()['price'].sort_values(ascending=False)
print(price_corr)


# In[37]:


data = data.drop('id', axis=1)
data = data.drop('zipcode',axis=1)


# In[38]:


# Feature Enginerring:
data['date'] = pd.to_datetime(data['date'])

data['month'] = data['date'].apply(lambda date:date.month)
data['year'] = data['date'].apply(lambda date:date.year)

data = data.drop('date',axis=1)

# Check the new columns
print(data.columns.values)


# In[39]:


data= data.drop('long',axis=1)


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


# Independent features:
X = data.drop('price',axis=1)


# Dependent features:
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:





# In[42]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[43]:


feature=SelectKBest(score_func=chi2,k=10)
fit=feature.fit(X,y)


# In[ ]:





# In[44]:


#import os
#X.to_csv(os.path.join(''), index=False)


# In[ ]:





# In[45]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[46]:


from sklearn.preprocessing import MinMaxScaler


# In[47]:


scaler = MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[48]:


from sklearn.linear_model import LinearRegression


# In[49]:


LR_score=[]
LR=LinearRegression()
LR.fit(X_train,y_train)
LR_score.append(LR.score(X_test,y_test))


# In[50]:


from sklearn.ensemble import RandomForestRegressor


# In[51]:


rf_score=[]
estimators=[1,2,3]
for R in estimators:
    rf_Regressor=RandomForestRegressor(n_estimators=R)
    rf_Regressor.fit(X_train,y_train)
    rf_score.append(rf_Regressor.score(X_test,y_test))


# In[52]:


from sklearn import ensemble


# In[53]:


GBR_score=[]
estimators=[10,20,30,40,50]
for G in estimators:
    GBR=ensemble.GradientBoostingRegressor(n_estimators = G)#, max_depth = 5, min_samples_split = 2,learning_rate = 0.1, loss = 'ls')
    GBR.fit(X_train, y_train)
    GBR_score.append(GBR.score(X_test,y_test))


# In[54]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[55]:


# Evaluation on Test data:
# predictions on the test set
predictions = rf_Regressor.predict(X_test)


# In[56]:


import joblib
joblib.dump(rf_Regressor,open('random.pkl','wb'))

#model_path = 'C:/Users/gts/House price predition using ML & NN/M3/random.pkl'
model = joblib.load('random.pkl')
#model = joblib.load(
    #open(model_path,'rb'))

#model = pickle.load(
    #open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')
@app.route('/about')    
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/result',methods=['POST'])
def predict():
    
    bedrooms = (request.form['bedrooms'])
    bathrooms = (request.form['bathrooms'])
    sqft_living = (request.form['sqft_living'])
    sqft_lot = (request.form['sqft_lot'])
    floors = (request.form['floors'])
   
    waterfront = (request.form['waterfront'])
    view = (request.form['view'])
    condition = (request.form['condition'])
    grade = (request.form['grade'])
    sqft_above = (request.form['sqft_above'])
    sqft_basement = (request.form['sqft_basement'])
    yr_built = (request.form['yr_built'])
    yr_renovated=(request.form['yr_renovated'])
    lat =(request.form['lat'])
    #long=(request.form['long'])
    sqft_living15 = (request.form['sqft_living15'])
    sqft_lot15 = (request.form['sqft_lot15'])
    month = (request.form['month'])
    year = (request.form['year'])
    
    #query = np.array([[bedrooms, bathrooms, sqft_living, sqft_lott, floors, waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,lat,sqft_living15,sqft_lot15, month, year]])
    query =np.array([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,lat,sqft_living15,sqft_lot15,month,year]])
#query=int(query)
    single_house = scaler.transform(query.reshape(-1, 18))
   
    
    prediction = model.predict(query)
    #prediction = round(prediction)
    print(prediction)
    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
