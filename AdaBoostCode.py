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

from WeightedWeakLinear import WeightedWeakLinearClassifier
import numpy as np

class AdaBoost:
    def __init__(self, n_learners=88):
        self.n_learners = n_learners
        self.learners = []
        self.learner_weights = []

    def fit(self, x, y):
        n_samples = x.shape[0]
        sample_weights = np.ones(n_samples) / n_samples  # initialize weights

        for _ in range(self.n_learners):
            # train a new weak learner
            learner = WeightedWeakLinearClassifier()
            learner.fit(x, y, sample_weights)
            # compute error and weight
            predictions = learner.predict(x)
            incorrect = predictions != y
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            # stop if error is too high or too low
            if error == 0:
                learner_weight = 1.0
            elif error >= 0.5:
                break
            else:
                learner_weight = 0.5 * np.log((1 - error) / error)

            sample_weights *= np.exp(-learner_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize

            # save learner/weight
            self.learners.append(learner)
            self.learner_weights.append(learner_weight)

    def predict(self, x):
        # predictions from all learners
        learner_preds = np.array([learner_weight * learner.predict(x)
                                  for learner, learner_weight in zip(self.learners, self.learner_weights)])
        return np.sign(np.sum(learner_preds, axis=0))