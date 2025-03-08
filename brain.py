import numpy as np


def get_empty_arr(nodes):
    """ok just wanna clarify. Returns a list of numpy matrices, each element in list representing a layer
    and each row of weights being each nodes wieghts so a struct of [2,3,1] will give back a 2 element list
    where the first element is a 3x2 matrix, again 1st row is the wieght between hidden layer 1 node1 and input1
    and W[0][0][1] being the weight between 1st hidden layer node1 and input2

    Bias b will be again a list where elements are hidden layers and each element is a column vctr
    where each row is the node in that layer, only one bias per node"""

    weights = []
    bias = []

    for idx in range(len(nodes)):
        try:
            arr = np.zeros((nodes[idx+1], nodes[idx]))
            single_bias = np.zeros((nodes[idx+1], 1))
        except IndexError:
            pass
        else:
            weights.append(arr)
            bias.append(single_bias)

    return weights, bias


def init_w_b(struct: list):
    w, b = get_empty_arr(struct)

    for idx in range(len(w)):
        w[idx] = w[idx] + \
            (np.random.rand(w[idx].shape[0],
                            w[idx].shape[1]) - 0.5)
        b[idx] = b[idx] + np.random.rand(b[idx].shape[0], 1) - 0.5

    return w, b


def train_every_pixel(loc, col, pixel_w, pixel_b):

    n = len(col)    # num pixels / inputs

    for i in range(len(col)):
        X, Y = np.array(loc[i]), np.array([col[i]])
        w, b = pixel_w[i], pixel_b[i]

        single_train(X, Y, w, b)


def single_train(X, Y, w, b):
    pass


def leaky_relu(x, alpha=0.05):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.05):
    return np.where(x > 0, 1, alpha)


def for_prop(inpts, weights, bias, act_func=0, out_func=1):

    funcs = [ReLU, sigmoid, SiLU, hyp_tan, softmax, leaky_relu]

    act_func = funcs[act_func]
    out_func = funcs[out_func]

    z = []
    a = []

    for layer_idx in range(len(weights)):

        if layer_idx == 0:
            dotted = inpts
        else:
            dotted = a[layer_idx-1]

        z.append(weights[layer_idx].dot(dotted) + bias[layer_idx])

        if layer_idx == len(weights) - 1:
            a.append(out_func(z[layer_idx]))
        else:
            a.append(act_func(z[layer_idx]))

    return z, a


def back_prop_func(inpts, z, a, w, Y, act_func=0):

    funcs = [der_ReLU, der_sigmoid, der_SiLU,
             der_hyp_tan, der_softmax, leaky_relu_derivative]
    der_func = funcs[act_func]
    examples = 1

    dw = [None for i in range(len(a))]
    dz = [None for i in range(len(a))]
    db = [None for i in range(len(a))]

    for layer_idx in reversed(range(len(a))):

        if layer_idx == len(a) - 1:
            dz[layer_idx] = 2 * (a[layer_idx] - Y)
            dw[layer_idx] = dz[layer_idx].dot(a[layer_idx-1].T) * (1/examples)
            db[layer_idx] = np.reshape(
                (np.sum(dz[layer_idx], axis=1) * (1/examples)).T, (a[layer_idx].shape[0], 1))
        else:
            if layer_idx == 0:
                dotted = inpts
            else:
                dotted = a[layer_idx - 1]

            dz[layer_idx] = \
                w[layer_idx + 1].T.dot(dz[layer_idx+1]) * \
                der_func(z[layer_idx])

            dw[layer_idx] = dz[layer_idx].dot(dotted.T) * (1/examples)
            db[layer_idx] = np.reshape(
                (np.sum(dz[layer_idx], axis=1) * (1/examples)).T, (a[layer_idx].shape[0], 1))

    return dw, db


def back_prop(inpts, z, a, w, Y):

    funcs = [der_ReLU, der_sigmoid, der_SiLU, der_hyp_tan, der_softmax]
    der_func = funcs[1]
    examples = 1

    dw = [None for i in range(len(a))]
    dz = [None for i in range(len(a))]
    db = [None for i in range(len(a))]

    for layer_idx in reversed(range(len(a))):

        dz[layer_idx] = 2 * (a[layer_idx] - Y)
        dw[layer_idx] = dz[layer_idx].dot(inpts.T) * (1/examples)
        db[layer_idx] = np.reshape(
            (np.sum(dz[layer_idx], axis=1) * (1/examples)).T, (a[layer_idx].shape[0], 1))

    return dw, db


def update_params(w, b, dw, db, lr):
    for layer_idx in range(len(w)):
        w[layer_idx] = w[layer_idx] - lr * dw[layer_idx]
        b[layer_idx] = b[layer_idx] - lr * db[layer_idx]

    return w, b


def ReLU(Z):
    return np.maximum(Z, 0)


def der_ReLU(A):
    return A > 0


def sigmoid(Z):
    return (1 / (1 + np.exp(-Z)))


def der_sigmoid(A):
    return sigmoid(A) * (1 - sigmoid(A))


def SiLU(Z):
    return Z * sigmoid(Z)


def der_SiLU(A):
    return (np.exp(A) + 1 + A * np.exp(A)) / (1 + np.exp(A))**2


def hyp_tan(Z):
    return np.tanh(Z)


def der_hyp_tan(A):
    return 1 - hyp_tan(A) * hyp_tan(A)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def der_softmax(A):
    return A
