import numpy as np

from neural_network import *

class TestPerceptron2in1out(object):
    n_in = 2
    n_out = 1
    cl = NeuralNetwork(n_in, n_out)

    def test_fit_n_in_1_out(self):
        X = np.array([[1,2],[3,4]])
        y = np.array([0,1])
        self.cl.fit(X,y)
        assert 1==1

    def test_network_init(self):
        self.cl = NeuralNetwork(self.n_in, self.n_out)
        assert self.cl.W.shape == (2,)

    def test_missing_arguments(self):
        # TODO: good way to do this?
        assert 1==1

    def test_sigmoid(self):
        assert self.cl._sigmoid_function(0) == 0.5

    def test_forward_prop(self):
        _cl = NeuralNetwork(self.n_in, self.n_out)
        _cl.W = np.array([1, -1])
        xd = np.array([2,2])
        assert _cl.propagate_forward(xd) == 0.5

    def test_grad_shape(self):
        xd = np.array([2, 2])
        yd = 1
        td = 0
        assert self.cl._compute_grad(td, yd, xd).shape == xd.shape

    def test_weight_update_shape(self):
        _cl = NeuralNetwork(self.n_in, self.n_out)
        _cl.W = np.array([1, 1])
        xd = np.array([2, 2])
        od = 1
        td = 0
        assert _cl._compute_weight_update(od, td, xd,_cl.W).shape == xd.shape

    # def test_sgd_update0(self):
    #     xd = np.array([100, 100])
    #     td = 0
    #     assert np.array_equal(self.cl._sgd_update(xd, td, self.cl.W),\
    #                           self.cl.W)
    #
    # def test_sgd_update1(self):
    #     xd = np.array([-100, -100])
    #     td = 1
    #     assert np.array_equal(self.cl._sgd_update(xd, td, self.cl.W),\
    #                           self.cl.W)

    def test_fit(self):
        n_examples = 100
        x1 = np.vstack([np.random.uniform(0, 5, n_examples / 2),
                        np.random.uniform(0, 5, n_examples / 2)])
        y1 = np.array([0] * (n_examples / 2))
        x2 = np.vstack([np.random.uniform(-5, 0, n_examples / 2),
                        np.random.uniform(-5, 0, n_examples / 2)])
        y2 = np.array([1] * (n_examples / 2))
        X = np.hstack([x1, x2])
        y = np.concatenate([y1, y2])
        self.cl.fit(X, y, n_iter=5)




