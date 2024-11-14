from WeightedWeakLinear import WeightedWeakLinearClassifier
import numpy as np

class AdaBoost:
    def __init__(self, n_learners=50):
        self.n_learners = n_learners
        self.learners = []
        self.learner_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples  # Initialize weights

        for _ in range(self.n_learners):
            # Train a new weak learner
            learner = WeightedWeakLinearClassifier()
            learner.fit(X, y, sample_weights)

            # Compute learner error and its weight
            predictions = learner.predict(X)
            incorrect = predictions != y
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # Stop if error is too high or too low
            if error == 0:
                learner_weight = 1.0
            elif error >= 0.5:
                break
            else:
                learner_weight = 0.5 * np.log((1 - error) / error)

            # Update sample weights
            sample_weights *= np.exp(-learner_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize

            # Save the learner and its weight
            self.learners.append(learner)
            self.learner_weights.append(learner_weight)

    def predict(self, X):
        # Aggregate predictions from all learners, weighted by learner weights
        learner_preds = np.array([learner_weight * learner.predict(X)
                                  for learner, learner_weight in zip(self.learners, self.learner_weights)])
        return np.sign(np.sum(learner_preds, axis=0))