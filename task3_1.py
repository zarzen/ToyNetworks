import pickle
import numpy as np
from task2_2 import shift_data

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10():
    data_folder = 'cifar-10-batches-py'
    train_batch = []
    for i in range(1,6):
        data = unpickle(data_folder+'/data_batch_'+str(i))
        train_batch.append(data)
    test_batch = unpickle(data_folder + '/test_batch')
    val_batch = train_batch[4]
    train_batch = train_batch[0:4]

    test_d = [shift_data(test_batch[b'data']), test_batch[b'labels']]
    val_d = [shift_data(val_batch[b'data']), val_batch[b'labels']]
    train_inputs = train_batch[0][b'data']
    train_labels = train_batch[0][b'labels']
    for b in train_batch[1:]:
        train_inputs = np.concatenate((train_inputs, b[b'data']), axis=0)
        train_labels.extend(b[b'labels'])
    train_labels = [vectorized_result(l) for l in train_labels]

    train_d = [shift_data(train_inputs), train_labels]

    # RGB2Gray
    test_d[0] = RGB2Gray(test_d[0])
    val_d[0] = RGB2Gray(val_d[0])
    train_d[0] = RGB2Gray(train_d[0])

    # reshape
    test_d[0] = [np.reshape(x, (1024,1)) for x in test_d[0]]
    val_d[0] = [np.reshape(x, (1024, 1)) for x in val_d[0]]
    train_d[0] = [np.reshape(x, (1024, 1)) for x in train_d[0]]

    # zip and list 
    train_d = list(zip(train_d[0], train_d[1]))
    val_d = list(zip(val_d[0], val_d[1]))
    test_d = list(zip(test_d[0], test_d[1]))

    return train_d, val_d, test_d


def vectorized_result(j):
    """ cifar 10 has 10 classes"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def RGB2Gray(data):
    """
    0.2126R + 0.7152G + 0.0722B
    """
    n = 32 * 32
    gray = np.zeros((len(data), n))
    gray += 0.2126 * data[:, 0:n]
    gray += 0.7152 * data[:, n:2*n]
    gray += 0.0722 * data[:, 2*n:3*n]

    return gray


def main():
    train, val, test = load_cifar10()
    train = train[0:5000]
    val = val[0:1000]
    test = test[0:1000]

    from task2_3 import Network
    network_sizes = [1024, 30, 10]
    mu = 0.9
    lr = 3
    lmd = 100.0
    print('momentum', mu, 'lambda', lmd)
    model = Network(network_sizes, momentum=mu, nesterov=True)
    model.SGD(train, 30, 100, lr, lmd, val)
    print('Evaluation on test {0} / {1}'.format(
        model.evaluate(test), len(test)))

if __name__ == "__main__":
    main()
