# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values and import linear regression from sklearn.
3. Assign the points for representing in the graph.
4. Predict the regression for marks by using the representation of the graph and compare the graphs and hence we obtained the linear regression for the given datas.
   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Abbu Rehan
RegisterNumber:  212223240165

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
# display the content in file
print(df.head())
print(df.tail())
# segeragating the values in data
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
# splitting train and test data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
# graph plot for training data
plt.scatter(x_train,y_train,color='orange')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# Mean Absolute Error
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![ml ex 02 01](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/8347df42-544e-475c-940d-be8da0c843d0)
![ml ex 02 02](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/5a0ee5e8-cb55-49f9-9b0e-2e1025deca7e)
![ml ex 02 03](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/f589ce6a-bd61-4b3a-beb3-d6a6703e5c89)
![ml ex 02 04](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/434f05af-531b-438c-9176-323900b74f7d)
![ml ex 02 05](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/f273b5a9-ff2a-4f9b-bfc0-cb2aa0bf38d7)
![ml ex 02 06](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/8c6dc082-7d96-4958-a57d-2ef9f629bb1e)
![ml ex 02 07](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/7cdc29b9-1c19-4829-9c1f-b5d8cf9e2fde)
![ml ex 02 08](https://github.com/Abburehan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849336/facc8a62-b01d-4f14-a229-7d44b0f78b57)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
