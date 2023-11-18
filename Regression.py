import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data=pd.read_csv('Stores.csv')
data.head()

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=11)

# Perform Stadradisation form scaling data to fit a standard normal distribution
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
# Stochasitic Gradient Descent
md1=SGDRegressor(alpha=0.01,max_iter=300)
md1.fit(X_train,y_train)
predl=md1.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,predl)))


# Custom Stochastic Gradient Descent
def CustomSGD(X_train,y_train,learning_rate,n_iter,batch_size):
  w=np.zeros(shape=(1,X_train.shape[1]))
  b=0
  cur_iter=1
  while(cur_iter <= n_iter):
    y=np.array(y_train)
    x=np.array(X_train)
    w_gradient=np.zeros(shape=(1,X_train.shape[1]))
    b_gradient=0

    for i in range(batch_size):
      prediction=np.dot(w,x[i])+b
      w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction))
      b_gradient=b_gradient+(-2)*(y[i]-(prediction))

    W = w-learning_rate*(w_gradient/batch_size)
    b = b-learning_rate*(b_gradient/batch_size)
    cur_iter=cur_iter+1
  return w,b