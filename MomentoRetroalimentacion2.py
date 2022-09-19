#Módulo 2. Momento de retroalimentación
#Implementación de una técnica de aprendizaje máquina con el uso de un framework

#Elaborado por: Grecia Pacheco Castellanos A01366730

from turtle import color
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import  GridSearchCV
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#MAIN

# Lectura de datos

df = pd.read_csv('train.csv')

df= df.drop([ 'PassengerId', 'Ticket', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked'], axis=1)

df.fillna(0)

x= df[['Pclass', 'Age']].to_numpy()
y= df[['Survived']].to_numpy()

x= np.nan_to_num(x)
y= np.concatenate(y, axis=None)

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2,random_state=1234)

# Entrenamiento de modelo



test_params = {
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000],
    'n_jobs' : [1,3,5,7]
}

params ={
    'C': 0.615848211066026,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 100,
    'n_jobs' : 1
    
}

rf_model = LogisticRegression(**params)

rf_model.fit(x_train,y_train)

#pprint(rf_model.get_params())

#grid_search = GridSearchCV(rf_model, param_grid=test_params)
#grid_search.fit(x_train, y_train)

#
# image.pngprint(grid_search.best_params_)



print("\n Train")
y_predicted=rf_model.predict(x_train)
print("score : " ,rf_model.score(x_train,y_train))
print("bias  : " ,rf_model.intercept_)
print(classification_report(y_train,y_predicted))
# Testing del modelo


print("\n Test")
y_predicted=rf_model.predict(x_test)
print("score : " ,rf_model.score(x_test,y_test))
print("bias  : " ,rf_model.intercept_)
print(classification_report(y_test,y_predicted))

new_x =[]
for i in range(0,len(y_test)):
    new_x.append(i)

fig,ax = plt.subplots()
ax.scatter( new_x, y_test)
ax.scatter(new_x,y_predicted, color= "red")
plt.show()
"""
p1 = max(max(y_test), max(y_predicted))
p2 = min(min(y_test), min(y_predicted))
plt.plot([p1, p2], [p1, p2])
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
"""

