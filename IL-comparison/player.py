import numpy as np

def normalize(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    m[np.isnan(m)] = 0.
    return m


class Player:
    def __init__(self,lexicon,epsilon):
        self.states = [00,01,10,11]
        self.messages = [00,01,10,11]
        self.lexicon = lexicon
        self.epsilon = epsilon

    def senderMatrix(self):
        out = np.zeros((len(self.states),len(self.messages)))
        for i in range(np.shape(out)[0]):
            for j in range(np.shape(out)[1]):
                if self.lexicon[i,j] == 1:
                    out[i,j] = 1 - self.epsilon
                else:
                    out[i,j] = self.epsilon / (len(self.messages) - 1)
        return out

    def receiverMatrix(self):
        return normalize(np.transpose(self.senderMatrix()))
