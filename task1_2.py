from mnist_loader import load_data_wrapper
from network import Network

def main():
    """"""
    train, val, test = load_data_wrapper()
    model = Network([784, 60, 10])
    model.SGD(train, 30, 10, 3, val)
    print('Evaluation on test: {0} / {1}'.format(model.evaluate(test), len(test)))

if __name__ == '__main__':
    main()
