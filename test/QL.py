import numpy as np

class QTable:
    def __init__(self) -> None:
        self.learned = dict()
        self.learning_rate = 1
        self.randomness = 1
    def test(self):
        pass

class Estimator:
    def __init__(self) -> None:
        self.record : list[np.ndarray] = [] # the list of board states and the captures

        pass
    def estimate(self, board: np.ndarray, captures, result):
        self.gamma = 0.0001
        if result is None:
            self.record.append((board, captures))
            return
        traceback = self.record.reverse
        traceback.reverse()
        for record in traceback:
            record
