# CSE 498 Deep Learning Homework2
# Task1
## 1.1
sigmoid function as activation function.

## 1.2
The output log is in the `task1_2_log.txt` file. 
The accuracy can achieve `95%`.

## 1.3 
The log of three experiments are in `task1_3_log.txt`, each experiment
result has been separated with comments. Run `task1_3.py` to reproduce experiment
result. We can see a large learning rate will make network perform very bad,
even compare with the base line, which does not have hidden layer.

## 1.4 
The log is in `task1_4_log.txt`. Run `task1_4.py` to reproduce experiment
result. We can see the accuracy on validation and test are close.

## 1.5
The `softmax` implementation is in the file `task1_5_softmax.py`. Where I copied
class `Network`, but modified the `backprop` function to calculate new `delta`
for `softmax`. The experiment output are included in `task1_5log.txt`. 

## 1.6
The implementation of `relu` is in `task1_6.py`, in which I replaced the `sigmoid`
function and its derivative. The `Network` class inside `task1_6.py` still 
using `softmax` as last layer output as `task1-5` did. But in order to make 
the network with `relu` function working good, we need to lower the learning
rate.

## 1.7 (Extra Credits)

The original ReLU activation function give zero gradient when its input is
negative. The origin ReLU will cause a lot dead neuron in networks. Because
there is no gradient to update weights if the output of the neuron is negative.
Leaky ReLU activation gives small gradient to negative output neurons. The
activation function is max(x, ax), where a is a small positive number.

# Task2
## 2.1
The network is converge quicker than origin random initialization. We can 
see the log in file `task2_1_log.txt`. After first epoch done, the accuracy 
on validation set achieves `94%`. Without better initialization the accuracy
is only `71%` after first epoch, based on the log output in `task1_2_log.txt`.

## 2.2 
The converge speed is faster than `non-preprocessed` data, because after 
first epoch iteration the accuracy can achieve `90%`. And we can see 
the learning curve is smoother than the learning curve `without preprocessing`. 
In the task1_2 output, `task1_2_log.txt`, we can see there is a big gap between
epoch `9` and epoch `10`. The accuracy increases almost `10%`. 

## 2.3
I have modified `back propagation` of in `task2_2.py` with momentum and nesterov
momentum. The experiments conducted in `main()` function. The log output is in 
`task2_3_log.txt`. In order to close look the effect of `momentum` SGD, I evaluate
network performance every 50 steps. I have separated three learning strategy but 
with same initialization. We can see `momentum SGD` and `nesterov momentum` converges
much faster then `vanilla` SGD. However, the large momentum cause instability 
during learning process. And large momentum with sigmoid activation also causes
`neuron saturating`. Thus, I have added `l2-regularization` for weights decay.


# Task3
## 3.2
The progress log and final evaluation is in file `task3_2_log.txt`. In order to make learning
faster, I am using `momentum SGD` implemented by `task2_3` and using part of `training data`

## 3.3
I combined three color channel into gray scale. The function `RGB2Gray` implementation
is in `task3_1.py`. The transformation is based on equation: 
`gray = 0.2126*R + 0.7152*G + 0.0722*B`
