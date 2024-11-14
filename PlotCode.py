import matplotlib.pyplot as plt
import numpy as np
from AdaBoostCode import AdaBoost

# Step 1: Load Data
# Load training and testing data
train_data = np.loadtxt('adaboost-train-24.txt')
test_data = np.loadtxt('adaboost-test-24.txt')

# Separate features (X) and labels (y)
X_train = train_data[:, :2]  # The first two columns are features
y_train = train_data[:, 2]  # The third column is the label

X_test = test_data[:, :2]
y_test = test_data[:, 2]

# Step 4: Train the AdaBoost classifier and plot accuracy
# Initialize and train AdaBoost
adaboost = AdaBoost(n_learners=50)
adaboost.fit(X_train, y_train)

# Track accuracy as we add more learners
train_accuracies = []
test_accuracies = []
for i in range(1, len(adaboost.learners) + 1):
    partial_ensemble = AdaBoost(n_learners=i)
    partial_ensemble.learners = adaboost.learners[:i]
    partial_ensemble.learner_weights = adaboost.learner_weights[:i]

    train_predictions = partial_ensemble.predict(X_train)
    test_predictions = partial_ensemble.predict(X_test)

    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot accuracy
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Learners')
plt.ylabel('Accuracy')
plt.title('AdaBoost Accuracy over Number of Learners')
plt.legend()
plt.show()
