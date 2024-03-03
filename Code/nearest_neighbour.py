from collections import Counter
from math import sqrt
from flatbuffers.builder import np

def euclidean(row1, row2):
    distance = [(row1[i] - row2[i])**2 for i in range(len(row1)-1)]
    return sqrt(sum(distance))

class knearestneighbour():
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.train = X
        self.target = Y

    def prediction(self, testx):
        result = []
        for tx in testx:
            dist = [euclidean(tx, x) for x in self.train]

            i = np.argsort(dist)[: self.k]

            labels = [self.target[j] for j in i]

            ans = Counter(labels).most_common(1)
            result.append(ans[0][0])
        return np.array(result)