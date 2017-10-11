import random
import numpy as np
from copy import deepcopy

class Network(object):

    def __init__(self, sizes, momentum=0.0, nesterov=False):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.momentum = momentum
        self.nesterov = nesterov

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmd=0.0,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        counter = 0
        # initialization for (t-1) velocity maintenance
        v_b_pre = [np.zeros(b.shape) for b in self.biases]
        v_w_pre = [np.zeros(w.shape) for w in self.weights]
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                v_b_pre, v_w_pre = self.update_mini_batch(mini_batch,
                                                          eta, lmd,
                                                          len(training_data),
                                                          v_b_pre, v_w_pre)
                counter+=1
                if counter % 50 == 0:
                    print("Step {0}: {1} / {2}".format(counter,
                                                       self.evaluate(test_data),
                                                       n_test))
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, lmd, n, v_b_pre, v_w_pre):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, v_b_pre, v_w_pre)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # # print gradients for debugging
        # for nw in nabla_w:
        #     print('gradients {0}'.format(np.linalg.norm(nw)))
        # update velocity
        v_b = list([self.momentum * v_pre - (eta/len(mini_batch)) * nb
                    for v_pre, nb in zip(v_b_pre, nabla_b)])
        v_w = list([self.momentum * v_pre - (eta/len(mini_batch)) * nw
                    for v_pre, nw in zip(v_w_pre, nabla_w)])
        # momentum update
        # if momentum is 0, it equals to vanilla update
        # DONE add l2 regularization: weights decay
        self.weights = [(1-eta*(lmd/n))*w + v
                        for w, v in zip(self.weights, v_w)]
        self.biases = [b + v for b, v in zip(self.biases, v_b)]

        return v_b, v_w

    def backprop(self, x, y, v_b_pre, v_w_pre):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        biases = self.biases
        weights = self.weights
        if self.nesterov:
            # use weights ahead of current weights, based on previous
            # velocity to step forward
            biases = [b + self.momentum * v_pre
                      for b, v_pre in zip(biases, v_b_pre)]
            weights = [w + self.momentum * v_pre
                       for w, v_pre in zip(weights, v_w_pre)]

        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def main():
    """"""
    from mnist_loader import load_data_wrapper
    train, val, test = load_data_wrapper()
    train = train[0:20000]
    val = val[0:1000]
    test = test[0:1000]

    n_epoch = 2
    network_sizes = [784, 30, 10]
    batch_size = 30
    lr = 3
    mu = 0.8
    lmd = 200.0
    print('batch_size {0}; learning rate {1}; momentum: {2}; lambda {3}'.format(
        batch_size, lr, mu, lmd
    ))

    # model1 vanilla updates
    print('='*20, 'vanilla update strategy', '='*20)
    model_vanilla = Network(network_sizes)
    init_biases = deepcopy(model_vanilla.biases)
    init_weights = deepcopy(model_vanilla.weights)
    model_vanilla.SGD(train, n_epoch, batch_size, lr, test_data=val)
    print('Evaluation on test: {0} / {1}'.format(model_vanilla.evaluate(test),
                                                len(test)))

    # model 2 standard momentum
    print('='*20, 'standard momentum', mu, '='*20)
    model_momentum = Network(network_sizes, momentum=mu)
    # guarantee same init weights
    model_momentum.biases = deepcopy(init_biases)
    model_momentum.weights = deepcopy(init_weights)
    model_momentum.SGD(train, n_epoch, batch_size, lr, lmd, val)
    print('Evaluation on test: {0} / {1}'.format(model_momentum.evaluate(test),
                                                 len(test)))

    # model 3 nesterov SGD
    print('='*20, 'nesterov SGD ', mu, '='*20)
    model_nesterov = Network(network_sizes, momentum=mu, nesterov=True)
    # guarantee same init weights
    model_nesterov.biases = deepcopy(init_biases)
    model_nesterov.weights = deepcopy(init_weights)
    model_nesterov.SGD(train, n_epoch, batch_size, lr, lmd, val)
    print('Evaluation on test: {0} / {1}'.format(model_nesterov.evaluate(test),
                                                 len(test)))
    
if __name__ == "__main__":
    main()
