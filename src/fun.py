import numpy as np

def apply_f(a, f):
    if isinstance(a, list):
        return list(map(lambda t: apply_f(t, f), a))
    else:
        return f(a)


def basic_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid(x):
    s = apply_f(x, lambda t: 1 / (1 + np.exp(-t)))
    return s


def sigmoid_derivative(x):
    ds = apply_f(x, lambda t: np.exp(-x) / (1 + np.exp(-x)) ** 2)
    return ds


def sigmoid_tan(x):
    s = apply_f(x, lambda t: (np.exp(t) - np.exp(-t)) / (np.exp(t) + np.exp(-t)))
    return s


def sigmoid_lin(x):
    s = apply_f(x, lambda t: t * 1)
    return s


def sigmoid_limit(x):
    a = dir(x)[0]
    if (a == "T"):
        s1 = lambda t: 1 if (t >= 0) else 0
        s2 = np.vectorize(s1)
        s = s2(x)
    else:
        s = apply_f(x, lambda t: 1 if (t >= 0) else 0)
    return s


def sigmoid_relu(x):
    a = dir(x)[0]
    if (a == "T"):
        s1 = lambda t: t if (t >= 0) else 0
        s2 = np.vectorize(s1)
        s = s2(x)
    else:
        s = apply_f(x, lambda t: t if (t >= 0) else 0)
    return s


def image_to_vector(image):
    v = image.ravel()
    v_1 = np.zeros(shape=(v.size, 1))
    for i in range(v.size):
        v_1[i, 0] = v[i]
    return v_1


def softmax(x):
    s = np.exp(x)
    s_new = np.zeros(shape=x.shape)
    shape_1 = x.shape[0]
    shape_2 = x.shape[1]
    for i in range(shape_1):
        s_1 = np.sum(s, axis=1)
        for j in range(shape_2):
            s_new[i, j] = s[i, j] / s_1[i]
    return s_new


def normalize(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_new = np.zeros(shape=x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_new[i, j] = x[i, j] / x_norm[i]
    return x_new



