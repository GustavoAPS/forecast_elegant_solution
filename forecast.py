import numpy as np
from sklearn.linear_model import LinearRegression

#Recebendo array de 24 valores para 24 horas
data = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1, 10.2, 11.3, 12.4, 13.5, 14.6, 15.7, 16.8, 17.9, 19.1, 20.2, 21.3, 22.4, 23.5, 24.6, 25.7])

#Reshape 
data = data.reshape(-1,1)

#Treinando o modelo usando e preparando para extracao de 6 valores
model = LinearRegression().fit(data[:-6], data[6:])

#prevendo
prediction = model.predict(data)

#Printando previsao
print("The temperatures for the next 6 hours are: ", prediction[-6:])