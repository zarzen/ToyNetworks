from mnist_loader import load_data_wrapper
from network import Network

def main():
    train, val, test = load_data_wrapper()
    # no hidden layer
    print('='*20, 'model: no hidden layer', '='*20)
    model1 = Network([784, 10])
    model1.SGD(train, 30, 10, 3, val)
    print('Evaluation on test: {0} / {1}'.format(model1.evaluate(test), len(test)))

    # small learning rate
    print('='*20, 'model: small learning rate', '='*20)
    model2 = Network([784, 30, 10])
    model2.SGD(train, 30, 10, 0.1, val)
    print('Evaluation on test: {0} / {1}'.format(model2.evaluate(test), len(test)))

    # large learning rate
    print('=' * 20, 'model: large learning rate', '=' * 20)
    model3 = Network([784, 30, 10])
    model3.SGD(train, 30, 10, 30, val)
    print('Evaluation on test: {0} / {1}'.format(model3.evaluate(test), len(test)))

if __name__ == '__main__':
    main()