import numpy as np
class DecisionTreeClassifier:
    def __init__(self):
        pass

    def compute_entropy(self, y):
        p1 = np.mean(y)

        if len(y) == 0 or p1 == 1 or p1 == 0:
            return 0
        
        p2 = 1 - p1
        entropy = np.dot(-p1, np.log2(p1)) + np.dot(-p2, np.log(p2))
        return entropy
    
        