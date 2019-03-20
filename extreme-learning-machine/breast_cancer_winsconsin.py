import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from elm import ELM
from mlxtend.plotting import plot_decision_regions


breast_cancer_dataset = datasets.load_breast_cancer()
X = breast_cancer_dataset.data
Y = breast_cancer_dataset.target
Y = Y.reshape((-1, 1))

accuracy = np.zeros((20, 1))
mean_time = 0
data = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80)
  
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    elm = ELM(hidden_units = 10, activation="log")
    #elm = ELM(hidden_units = 10, activation="tan")
    elm.fit(X_train, Y_train)
    Y_hat = elm.predict(X_test)
    Y_hat = np.round(Y_hat)  

    accuracy[i] = 1 - np.sum(np.abs(np.sum(Y_hat - Y_test))) / Y_hat.size
    data.append([X_train, X_test, Y_train, Y_test, Y_hat])

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))
X_train, X_test, Y_train, Y_test, Y_hat = data[(np.abs(accuracy - np.mean(accuracy))).argmin()]


conf_matrix = confusion_matrix(Y_test, Y_hat)
print("Confusion Matrix")
print(conf_matrix)

labels = ["Benigno", "Maligno"]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title("Confusion Matrix")
fig.colorbar(cax)
ax.set_xticklabels([""] + labels)
ax.set_yticklabels([""] + labels)
plt.xlabel("Predicted")
plt.ylabel("Desired")
plt.show()
