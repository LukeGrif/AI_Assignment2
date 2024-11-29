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

import numpy as np

# WeightedWeakLinearClassifier class
class WeightedWeakLinearClassifier:
    def __init__(self):
        self.threshold = None
        self.direction = None

    def fit(self, x, y, sample_weights):
        # weighted mean
        positive_class = x[y == 1]
        negative_class = x[y == -1]
        positive_weight = sample_weights[y == 1]
        negative_weight = sample_weights[y == -1]

        mean_pos = np.average(positive_class, axis=0, weights=positive_weight)
        mean_neg = np.average(negative_class, axis=0, weights=negative_weight)
        # difference between the two means
        self.direction = mean_pos - mean_neg
        self.direction /= np.linalg.norm(self.direction)

        # all points projected
        projections = x @ self.direction
        sorted_indices = np.argsort(projections)

        min_error = float('inf')    # find best point
        for i in range(1, len(projections)):
            threshold = (projections[sorted_indices[i - 1]] + projections[sorted_indices[i]]) / 2
            predictions = np.where(projections >= threshold, 1, -1)
            error = np.sum(sample_weights[predictions != y])
            if error < min_error:
                min_error = error
                self.threshold = threshold

    def predict(self, x):
        projections = x @ self.direction
        return np.where(projections >= self.threshold, 1, -1)

