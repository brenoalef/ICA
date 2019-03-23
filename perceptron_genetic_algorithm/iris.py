import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from perceptron_ga import PerceptronGA
import time
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix


iris_datasets = datasets.load_iris()
X = iris_datasets.data
Y = np.zeros((iris_datasets.target.shape[0], 1))
Y[iris_datasets.target != 0] = 0
Y[iris_datasets.target == 0] = 1

accuracy = np.zeros((20, 1))
data = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
      
    perceptron_ga = PerceptronGA()
    perceptron_ga.fit(X_train, Y_train)
    Y_hat = perceptron_ga.predict(X_test)
    Y_hat = np.round(Y_hat)  

    accuracy[i] = np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test)
    data.append([X_train, X_test, Y_train, Y_test, Y_hat])

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))
X_train, X_test, Y_train, Y_test, Y_hat = data[accuracy.argmax()]


conf_matrix = confusion_matrix(Y_test, Y_hat)
print("Confusion Matrix")
print(conf_matrix)

labels = ["Setosa", "Other"]
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix)
ax.set_xticklabels([""] + labels)
ax.set_yticklabels([""] + labels)
plt.xlabel("Predicted")
plt.ylabel("Desired")
plt.show()