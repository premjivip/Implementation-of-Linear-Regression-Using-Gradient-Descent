# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
# Step 1 :

Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the Gradient Descent.
# Step 2 :

Upload the dataset conditions and check for any null value in the values provided using the .isnull() function.
# Step 3 :

Declare the default values such as n, m, c, L for the implementation of linear regression using gradient descent.
# Step 4 :

Calculate the loss using Mean Square Error formula and declare the variables y_pred, dm, dc to find the value of m.
# Step 5 :

Predict the value of y and also print the values of m and c.
# Step 6 :

Plot the accquired graph with respect to hours and scores using the scatter plot function.
# Step 7 :
End the program.

## Program:

/*
Program to implement the linear regression using gradient descent.
Developed by: premji p
RegisterNumber: 212221043004
*/


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function in a linear regression model

  """
  m=len(y)  
  h=X.dot(theta)
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  """
   Take in numpy array X,y and theta and update theta by taking number with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

## Output:
# Profit Prediction graph
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/9420b984-95ec-4511-94f3-6cfe6de24cdf)

# Compute Cost Value
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/281c0ca6-2adf-41f8-a4f7-29d40ceebef8)

# h(x) Value
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/571f455d-598a-43ac-a56e-bd491d87d16d)

# Cost function using Gradient Descent Graph
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/a8c2100c-c1cc-45c0-9011-8bb1419fe745)

# Profit Prediction Graph
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/2a53de61-30e2-4618-bcc2-fbd0fab9caa8)

# Profit for the Population 35,000
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/384629e7-4ec6-430e-af64-14cc73366015)

# Profit for the Population 70,000
![image](https://github.com/Yogabharathi3/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118899387/c7ce01d7-a54a-485d-ab17-8206b666896f)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
