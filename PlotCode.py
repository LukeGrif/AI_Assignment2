"""
AI - Project Two
Assignment 1
Group Members:
    - Luke Griffin      21334528
    - Taha AL-Salihi    21302227
    - Patrick Crotty    21336113
    - Eoin O'Brien      21322902
    - Mark Griffin      20260229

Description:
This code implements a ADABOOST CLASSIFIER USING ‘WEIGHTED WEAK LINEAR’ BASE CLASSIFIERS which achieves 100% Training
Accuracy and 97% Testing Accuracy
"""

import matplotlib.pyplot as plt
import numpy as np
from AdaBoostCode import AdaBoost

# load the data
train_data = np.loadtxt('adaboost-train-24.txt')
test_data = np.loadtxt('adaboost-test-24.txt')

# convert to x and y values
X_train, y_train = train_data[:, :2], train_data[:, 2]
X_test, y_test = test_data[:, :2], test_data[:, 2]

# 88 learners is optimum performance from trial and error
adaboost = AdaBoost(n_learners=88)
adaboost.fit(X_train, y_train)
train_accuracies = []
test_accuracies = []

for num_learners in range(1, len(adaboost.learners) + 1):
    partial_model = AdaBoost(n_learners=num_learners)
    partial_model.learners = adaboost.learners[:num_learners]
    partial_model.learner_weights = adaboost.learner_weights[:num_learners]
    train_prediction = partial_model.predict(X_train)
    test_prediction = partial_model.predict(X_test)
    train_accuracies.append(np.mean(train_prediction == y_train))
    test_accuracies.append(np.mean(test_prediction == y_test))

# plot fitted training and testing values
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Learners')
plt.ylabel('Accuracy')
plt.title('AdaBoost Accuracy with Increasing Learners')
plt.legend()
plt.grid(True)

# when plot exited grid plot appears
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = adaboost.predict(grid_points).reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')  # Decision regions
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', edgecolor='k', s=100)
plt.colorbar(label='Predicted Label')
plt.xlabel('Data 1')
plt.ylabel('Data 2')
plt.title('AdaBoost Decision Boundary')
plt.grid(True)

plt.show()
