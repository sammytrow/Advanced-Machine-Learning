import numpy as np
import matplotlib.pyplot as plt

class SupportVector:
    def __init__(self, iterations, lr, lam):
        self.weight = None
        self.bias = None
        self.iterations = iterations
        self.learning = lr
        self.lamb = lam
        self.errors= []

    def train(self, train_data, target):
        n_samples, n_features = train_data.shape

        #y_ = np.where(target <= 0, -1, 1)
        y_ = target

        self.weight = np.zeros(n_features)
        self.bias = 0

        for epoch in range(1, self.iterations):
            error = 0
            for i, x in enumerate(train_data):
                if (target[i] * (np.dot(x, self.weight) - self.bias) >= 1):
                    self.weight -= self.learning * (2 * self.lamb * self.weight)
                else:
                    self.weight -= self.learning * (
                            2 * self.lamb * self.weight - np.dot(x, y_[i])
                    )
                    self.bias -= self.learning * y_[i]
                    error += 1
            self.errors.append(error)

    def prediction(self, test_data):
        result = np.dot(test_data, self.weight) - self.bias
        return np.sign(result)

    def plot_results(self, X, plot_title):
        error_rate = [i / len(X) for i in self.errors]
        plt.plot(error_rate)
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.title(plot_title)
        filepath = "plots/" + plot_title.replace(" ", "_") + ".jpg"
        plt.savefig(filepath)
        plt.show()

