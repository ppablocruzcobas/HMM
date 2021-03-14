
__authors__ = "Pedro Pablo"


import HiddenMarkovModel as hmm
import Viterbi as v


if __name__ == "__main__":
    alpha = .1
    r1 = .9
    r2 = .5

    A = [[(1 - alpha) ** 2, 2 * alpha * (1 - alpha), alpha ** 2],
         [0, 1 - alpha, alpha], [0, 0, 1]]
    B = [[r1 ** 2, 2 * r1 * (1 - r1), (1 - r1) ** 2],
         [r1 * r2, r1 * (1 - r2) + r2 * (1 - r1), (1 - r1) * (1 - r2)],
         [r2 ** 2, 2 * r2 * (1 - r2), (1 - r2) ** 2]]
    # The initial state is 0.
    p = [1, 0, 0]

    H = hmm.HiddenMarkovModel(A, B, p)

    L = 17
    S = H.generate_signal(L)  # A random generated S6
    print("S = %s" % S)

    print(25 * "-")
    P0 = H.calculate(a={L: 0}, signal=S)
    P1 = H.calculate(a={L: 1}, signal=S)
    P2 = H.calculate(a={L: 2}, signal=S)
    print("P0 = P(x%s=0 | S%s) = %0.8f" % (L, L, P0))
    print("P1 = P(x%s=1 | S%s) = %0.8f" % (L, L, P1))
    print("P2 = P(x%s=2 | S%s) = %0.8f" % (L, L, P2))
    print("")
    print("P0 + P1 + P2 = %0.8f" % (P0 + P1 + P2))

    print(25 * "-")
    P0 = H.calculate(a={L + 1: 0}, signal=S)
    P1 = H.calculate(a={L + 1: 1}, signal=S)
    P2 = H.calculate(a={L + 1: 2}, signal=S)
    print("P0 = P(x%s=0 | S%s) = %0.8f" % (L + 1, L, P0))
    print("P1 = P(x%s=1 | S%s) = %0.8f" % (L + 1, L, P1))
    print("P2 = P(x%s=2 | S%s) = %0.8f" % (L + 1, L, P2))
    print("")
    print("P0 + P1 + P2 = %0.8f" % (P0 + P1 + P2))

    print(25 * "-")
    P0 = H.calculate(a={L + 1: 0}, signal=S, ameans='signal')
    P1 = H.calculate(a={L + 1: 1}, signal=S, ameans='signal')
    P2 = H.calculate(a={L + 1: 2}, signal=S, ameans='signal')
    print("P0 = P(s%s=0 | S%s) = %0.8f" % (L + 1, L, P0))
    print("P1 = P(s%s=1 | S%s) = %0.8f" % (L + 1, L, P1))
    print("P2 = P(s%s=2 | S%s) = %0.8f" % (L + 1, L, P2))
    print("")
    print("P0 + P1 + P2 = %0.8f" % (P0 + P1 + P2))

    print(25 * "-")
    print("P(S%s) = %0.8f" % (L, H.calculate(signal=S)))

    # Viterbi
    print(25 * "-")
    print("Viterbi algorithm")
    X, P = v.viterbi(A, B, p, S)
    print("Argmax P(X | S) = ", X, ". P = %0.8f" % P)
