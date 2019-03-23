import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from elm import ELM
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix


dataset = np.genfromtxt('dermatology.data', delimiter=',')
dataset = dataset[~np.isnan(dataset).any(axis=1), 0:35]
X = dataset[:, 0:-2]
Y = np.zeros((dataset.shape[0], 6))
for i in range(X.shape[0]):
    Y[i, int(dataset[i, -1]) - 1] = 1

iters = 20
n_folds = 3
accuracy = np.zeros((20, 1))
mean_time = 0
data = []
best = [[], 0]
for i in range(iters):
    CVO = KFold(n_splits=n_folds, shuffle=True)
    acc_values = []
    for train_index, test_index in CVO.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
  
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        elm = ELM(hidden_units = 50, activation="log")
        #elm = ELM(hidden_units = 50, activation="tan")
        elm.fit(X_train, Y_train)
        Y_hat = elm.predict(X_test)
        Y_hat = np.round(Y_hat)  
        acc_values.append(np.sum(np.where(np.argmax(Y_hat, axis=1) == np.argmax(Y_test, axis=1), 1, 0))/len(Y_test))
        if acc_values[-1] > best[1]:
            best[0] = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_hat, axis=1))
            best[1] = acc_values[-1]
    accuracy[i] = np.mean(acc_values)

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

conf_matrix = best[0]
print("Confusion Matrix")
print(conf_matrix)

labels = ["Psoriasis", "Seboreic Dermatitis", "Lichen Planus", "Pityriasis Rosea", "Cronic Dermatitis", "Pityriasis Rubra Pilaris "]
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix)
plt.xlabel("Predicted")
plt.ylabel("Desired")
plt.show()
