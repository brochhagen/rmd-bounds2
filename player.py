import numpy as np

def normalize(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    m[np.isnan(m)] = 0.
    return m

class LiteralPlayer:
    def __init__(self,alpha,lexicon):
        self.alpha = alpha
        self.lexicon = lexicon
        self.sender_matrix = self.sender_selection_matrix(lexicon)
        self.receiver_matrix =  self.receiver_selection_matrix()
   
    def sender_selection_matrix(self,l):
        #Compute e^([utility(column-cell given row-cell]**alpha)
        m = np.zeros(np.shape(l))
        for i in range(np.shape(l)[0]):
            for j in range(np.shape(l)[1]):
                m[i,j] = np.exp(np.power(l[i,j],self.alpha))
        return normalize(m)

    def receiver_selection_matrix(self):
        """Take transposed lexicon and normalize row-wise (prior over states plays no role as it's currently uniform)"""
        m = normalize(np.transpose(self.lexicon))
        for r in range(np.shape(m)[0]):
            if sum(m[r]) == 0:
                for c in range(np.shape(m)[1]):
                    m[r,c] = 1. / np.shape(m)[0]

        return m



#######################################
class GriceanPlayer:
    def __init__(self,alpha, lam, lexicon):
        self.alpha = alpha
        self.lam = lam
        self.lexicon = lexicon
        self.sender_matrix = self.sender_selection_matrix(lexicon)
        self.receiver_matrix = self.receiver_selection_matrix()

    def sender_selection_matrix(self,l):
        literalListener = normalize(np.transpose(l))
        utils = np.transpose(literalListener)
        return normalize(np.exp(self.lam*np.power(utils, self.alpha)))

    def receiver_selection_matrix(self):
        """Take transposed sender matrix and normalize row-wise (prior over states plays no role as it's currently uniform)"""
        return normalize(np.transpose(self.sender_matrix))
