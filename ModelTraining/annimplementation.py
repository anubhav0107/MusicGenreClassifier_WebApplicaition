import pandas as pd
import librosa as lb
import numpy as np

dataset = pd.read_csv('MFCCValues3csv.csv')

X = dataset.iloc[:, 1:19].values
Y = dataset.iloc[:, 19].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 30, epochs = 200)

Y_pred = classifier.predict(X_test)

test_loss, test_acc = classifier.evaluate(X_test,y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
