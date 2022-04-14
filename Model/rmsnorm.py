

class RMSNorm:

    def __init__(self, eps = 1e-8, p=-1., bias=False):
        self.eps = eps
        self.p = p
        self.bias = bias


    def call(self, x):

