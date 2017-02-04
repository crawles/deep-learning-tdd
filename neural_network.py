import numpy as np

class NeuralNetwork(object):
    '''
    Implement neural networks. Implementation guided by Mitchell 1997.

    '''

    def __init__(self, n_in, n_out, n_hidden=0, learning_rate=0.05):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.W = self.init_weights()

    def init_weights(self):
        W_dim = self.n_in if self.n_out == 1 else (self.n_in, self.out_dim)
        return np.random.uniform(0, .05, W_dim)

    def _apply_weights(self, xd):
        return np.dot(self.W, xd)

    def _sigmoid_function(self, y):
        print y
        return (1/(1 + np.exp(y)))

    def propagate_forward(self, xi):
        netj = self._apply_weights(xi)
        oj = self._sigmoid_function(netj)
        return(oj)

    def _compute_grad(self, oj, tj, xi):
        '''
        Compute the weight gradient for layer i to layer j

        :param oj: output at unit j
        :param tj: target output at unit j
        :param xi: input at unit i
        :return:
        '''
        return np.dot(-oj*(1-oj)*(tj-oj), xi)

    def _compute_weight_update(self, oj, tj, xi, wji):
        '''
        Updates the weights from layer i to layer j

        :param oj: output at unit j
        :param tj: target output at unit j
        :param xi: input at unit i
        :return:
        '''
        grad = self._compute_grad(oj, tj, xi)
        update = self.learning_rate * grad
        return(wji+update)

    def _sgd_update(self, xi, tj, wji):
        '''
        Stochastic gradient descent (SGD) update
        :param xd:
        :param yd:
        :return:
        '''
        od = self.propagate_forward(xi)

        delw = self._compute_weight_update(od, tj, xi, wji)
        return wji + delw

    def fit(self,X, y, n_iter = 10):
        for i in xrange(n_iter):
            for j in xrange(X.shape[1]):
                xj = X[:,j]
                yj = y[j]
                delw = self._sgd_update(xj,yj,self.W)
                self.W += delw
