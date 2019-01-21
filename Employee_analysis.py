#Artificial Neural Netwrok.

#importing libraries.
import pandas as pd

"""                 Data Preprocessing                     """

#importing the data.
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3: 13]
Y = dataset.iloc[:, 13]

#encoding categorical data.
X = pd.get_dummies(X, columns = ['Geography'], prefix = ['Country'])            # This method is called One-Hot encoding.
X = X.iloc[:, :-1]              #avoiding dummy variable trap.

#enooding binary data.
from sklearn.preprocessing import LabelEncoder
classifier_encoder = LabelEncoder()
X["Gender"] = classifier_encoder.fit_transform(X["Gender"])

#feature scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = X.astype('float')
X = sc.fit_transform(X)

#splitting data.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

No_of_features = len(X_train[0])

"""                 Constructing ANN                        """

#importing keras library.
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN.
classifier = Sequential()

#adding the input layer and the first hidden layer.
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = No_of_features))        # 6 = (1+11)/ 2

#adding the second hidden layer.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#adding the output layer.
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))             #soft_max function if 2 or more dependent features.

#compiling the ANN.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the trainging data to ANN.
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

"""                 Making Predictions                        """

#predicting test results.
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#making the confusion matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#calculating accuracy.
num = 0
dem = 0
for i in range(len(cm)):
    for j in range(len(cm)):
        if i == j:
            num = num + cm[i][j]
        dem = dem + cm[i][j]
print("Total accuracy: " + str(float(num/dem)))
