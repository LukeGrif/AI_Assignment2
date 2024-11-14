import numpy as np


# Step 2: Define the WeightedWeakLinearClassifier class
class WeightedWeakLinearClassifier:
    def __init__(self):
        self.threshold = None
        self.direction = None

    def fit(self, X, y, sample_weights):
        # Calculate the weighted mean for each class
        positive_class = X[y == 1]
        negative_class = X[y == -1]

        positive_weight = sample_weights[y == 1]
        negative_weight = sample_weights[y == -1]

        mean_pos = np.average(positive_class, axis=0, weights=positive_weight)
        mean_neg = np.average(negative_class, axis=0, weights=negative_weight)

        # Orientation vector between the two means
        self.direction = mean_pos - mean_neg
        self.direction /= np.linalg.norm(self.direction)

        # Project all points onto the direction vector
        projections = X @ self.direction
        sorted_indices = np.argsort(projections)

        # Find best split point
        min_error = float('inf')
        for i in range(1, len(projections)):
            threshold = (projections[sorted_indices[i - 1]] + projections[sorted_indices[i]]) / 2
            predictions = np.where(projections >= threshold, 1, -1)
            error = np.sum(sample_weights[predictions != y])

            if error < min_error:
                min_error = error
                self.threshold = threshold

    def predict(self, X):
        # Project onto the chosen direction and apply threshold
        projections = X @ self.direction
        return np.where(projections >= self.threshold, 1, -1)

