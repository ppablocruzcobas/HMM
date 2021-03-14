# Hidden Markov Model class

__authors__ = "Pedro Pablo"


import numpy as np


class HiddenMarkovModel():

    def __init__(self, A, B, p):
        self.signal = []
        self.alphas = []
        self.S = 0.0

        self.A = A
        self.B = B

        self.p = p

        self.n, self.m = np.shape(B)

    def __probability(self, vector):
        u = np.random.random()
        s = .0
        for i in range(len(vector) - 1):
            s = s + vector[i]
            if u <= s:
                return i
        return len(vector) - 1

    def generate_signal(self, length, state=None):
        if state == None:
            state = self.__probability(self.p)
        signal = [self.__probability(self.B[state])]
        for _ in range(length - 1):
            state = self.__probability(self.A[state])
            signal.append(self.__probability(self.B[state]))
        return signal

    def alpha(self, n, j):
        assert type(n) == int and n >= 0, "n must be an integer >= 0!"
        if n == 0:
            return self.B[j][self.signal[0]] * self.p[j]
        elif n >= 1:
            return self.B[j][self.signal[n]] * sum(A[j] * self.alpha(n - 1, i)
                for i, A in enumerate(self.A))

    def calculate(self, a=None, signal=[], ameans='state'):
        """
        `a` is a value:
        `b` is a list of signals.
        `ameans` stands for what is `a` representing ('signal' or 'state')
        returns P(a|b) if a is not None, otherwise P(b).
        """
        l = len(signal)
        # While the vector signal is the same, do not re-calculate `alphas`
        if signal != self.signal:
            self.signal = signal
            self.alphas.clear()
            for j in range(self.m):
                self.alphas.append(self.alpha(l - 1, j))
            self.S = sum(self.alphas)
            # print(self.alphas)

        if a is not None:
            key = list(a.keys())[0]
            value = list(a.values())[0]
            if ameans == 'state':

                # P(Xn=value | Sn=signal)
                if list(a.keys())[0] == l:
                    return self.alphas[value] / self.S

                # P(Xn+1=value | Sn=signal)
                elif key == l + 1:
                    return sum(self.A[i][value] * self.alphas[i] for i in range(self.n)) / self.S

            # P(Sn+1=value | Sn=signal)
            elif ameans == 'signal' and key == l + 1:
                return sum(self.B[i][value]
                           * self.calculate(a={key: i}, signal=signal)
                           for i in range(self.m))

        # P(Sn = signal)
        else:
            return self.S
