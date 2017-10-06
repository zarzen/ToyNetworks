# reuse
from network import Network
from mnist_loader import load_data
from mnist_loader import vectorized_result
import numpy as np


def load_data_wrapper():
    """wrapping data for network input"""
    tr_d, va_d, te_d = load_data()
    tr_d = (shift_data(tr_d[0]), tr_d[1])
    va_d = (shift_data(va_d[0]), va_d[1])
    te_d = (shift_data(te_d[0]), te_d[1])
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def shift_data(x):
    """"""
    mean_of_x = np.mean(x)
    x = x - mean_of_x

    std = np.std(x)
    x = x / std
    return x

def main():
    """main logic"""
    # this load_data_wrapper function is modified version
    train, val, test = load_data_wrapper()

    model = Network([784, 30, 10])
    model.SGD(train, 30, 10, 3, val)
    print('Evaluation on test: {0} / {1}'.format(model.evaluate(test), len(test)))

if __name__ == "__main__":
    main()
