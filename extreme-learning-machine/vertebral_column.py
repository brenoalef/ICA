import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from elm import ELM
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix


dataset = np.genfromtxt("vertebral_column.data", delimiter=",")
#dataset = dataset[~np.isnan(dataset).any(axis=1), 0:6]
X = dataset[:, 0:6]
Y = dataset[:, 6]
Y = np.zeros((dataset.shape[0], 3))
for i in range(Y.shape[0]):
    Y[i, int(dataset[i, -1])] = 1

accuracy = np.zeros((20, 1))
data = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
      
    elm = ELM(hidden_units = 20, activation="log")
    #elm = ELM(hidden_units = 20, activation="tan")
    elm.fit(X_train, Y_train)
    Y_hat = elm.predict(X_test)
    Y_hat = np.round(Y_hat)  

    accuracy[i] = np.sum(np.where(np.argmax(Y_hat, axis=1) == np.argmax(Y_test, axis=1), 1, 0))/len(Y_test)
    data.append([X_train, X_test, Y_train, Y_test, Y_hat])

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))
X_train, X_test, Y_train, Y_test, Y_hat = data[accuracy.argmax()]


conf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_hat, axis=1))
print("Confusion Matrix")
print(conf_matrix)

labels = ["Normal", "Disk Hernia", "Spondilolysthesis"]
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix)
ax.set_xticklabels([""] + labels)
ax.set_yticklabels([""] + labels)
plt.xlabel("Predicted")
plt.ylabel("Desired")
plt.show()