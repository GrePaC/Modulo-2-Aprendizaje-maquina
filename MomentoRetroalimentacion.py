#Módulo 2. Momento de retroalimentación
#Implementación de una técnica de aprendizaje máquina sin el uso de un framework

#Elaborado por: Grecia Pacheco Castellanos A01366730

import pandas as pd
import numpy as np
import math


#Implementación de regresión logística

class RegresionLogistica:

    def __init__(self, rate, iterations):
        self.rate = rate
        self.itr = iterations
        self.weights = None
        self.bias = None


    def fun_sigmoide(self,x):
        r = 1 / (1 + np.exp(-x))
        return r

    def fitting(self, x, y):
        samples, features = x.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.itr):

            linear_m = np.dot(x, self.weights) + self.bias

            y_expected = self.fun_sigmoide(linear_m)

            
            derivative_w = (1/ samples) * np.dot(x.T , (y_expected -y))
            #derivative_w = (1/ samples) *(np.mat(x) * np.mat(y))
            
            derivative_b = (1/ samples) * np.sum(y_expected - y)

            self.weights -= self.rate * derivative_w
            self.bias = self.rate * derivative_b

    def predictions(self, x):
        linear_m = np.dot(x, self.weights) + self.bias
        y_expected = self.fun_sigmoide(linear_m)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_expected]
        return np.array(y_predicted_cls)

def check_accuracy(expected, real):
    acc = np.sum(real == expected)/ len(real)
    return acc


#MAIN



#Importación de datos

df = pd.read_csv('train.csv')

df= df.drop([ 'PassengerId', 'Ticket', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked'], axis=1)

df.fillna(0)

x= df[['Pclass', 'Age']].to_numpy()
y= df[['Survived']].to_numpy()

x= np.nan_to_num(x)
y= np.concatenate(y, axis=None)

#División de dataset 

pr= 0.8
n_train = math.floor(pr * x.shape[0])
n_test = math.ceil((1-pr) * x.shape[0])
x_train = x[:n_train]
y_train = y[:n_train]
x_test = x[n_train:]
y_test = y[n_train:]

# Uso de regresión logística

log_reg = RegresionLogistica(0.0001, 100)
log_reg.fitting(x_train, y_train)


# Testing del modelo

print("\n Train")
y_predicted=log_reg.predictions(x_train)
print("score : " ,check_accuracy(y_train,y_predicted))
print("bias  : " ,log_reg.bias)

print("\n Test")
y_predicted=log_reg.predictions(x_test)
print("score : " ,check_accuracy(y_test,y_predicted))
print("bias  : " ,log_reg.bias)