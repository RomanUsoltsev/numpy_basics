import matplotlib.pyplot as plt
import pylab

import fun


def test_basic_sigmoid():
    assert round(fun.basic_sigmoid(3), 2) == 0.95
    print("test_basic_sigmoid")


def test_sigmoid():
    x = fun.np.array([1, 2, 3])
    res = fun.sigmoid(x)
    assert round(res[0], 2) == 0.73
    assert round(res[1], 2) == 0.88
    assert round(res[2], 2) == 0.95
    print("test_sigmoid")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    x = fun.np.linspace(-10, 10, 100)
    plt.plot(x, fun.sigmoid(x))
    plt.show()


def test_sigmoid_derivative():
    x = fun.np.array([1, 2, 3])
    res = fun.sigmoid_derivative(x)
    assert round(res[0], 2) == 0.20
    assert round(res[1], 2) == 0.10
    assert round(res[2], 3) == 0.045
    print("test_sigmoid_derivative")


def test_sigmoid_fun():
    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 2, 1)
    ax_2 = fig.add_subplot(2, 2, 2)
    ax_3 = fig.add_subplot(2, 2, 3)
    ax_4 = fig.add_subplot(2, 2, 4)

    ax_1.spines['left'].set_position('center')
    ax_1.spines['bottom'].set_position('center')

    ax_2.spines['left'].set_position('center')
    ax_2.spines['bottom'].set_position('center')

    ax_3.spines['left'].set_position('center')
    ax_3.spines['bottom'].set_position('center')

    ax_4.spines['left'].set_position('center')
    ax_4.spines['bottom'].set_position('center')

    x = fun.np.linspace(-10, 10, 100)
    pylab.subplot(2, 2, 1)
    pylab.plot(x, fun.sigmoid_relu(x))
    pylab.title("sigmoid_relu")

    pylab.subplot(2, 2, 2)
    pylab.plot(x, fun.sigmoid_tan(x))
    pylab.title("sigmoid_tan")

    pylab.subplot(2, 2, 3)
    pylab.plot(x, fun.sigmoid_lin(x))
    pylab.title("sigmoid_lin")

    pylab.subplot(2, 2, 4)
    pylab.plot(x, fun.sigmoid_limit(x))
    pylab.title("sigmoid_limit")

    plt.show()


def test_image_to_vector():
    image = fun.np.array([
        [[0.67826139, 0.29380381],
         [0.90714982, 0.52835647],
         [0.4215251, 0.45017551]],

        [[0.92814219, 0.96677647],
         [0.85304703, 0.52351845],
         [0.19981397, 0.27417313]],

        [[0.60659855, 0.00533165],
         [0.10820313, 0.49978937],
         [0.34144279, 0.94630077]]
    ])
    res = fun.image_to_vector(image)
    assert round(res[5, 0], 2) == 0.45
    assert round(res[8, 0], 2) == 0.85
    assert round(res[2, 0], 2) == 0.91
    print("test_image_to_vector")


def test_softmax():
    x = fun.np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    res = fun.softmax(x)
    assert round(res[1, 0], 2) == 0.88
    assert round(res[0, 3], 5) == 0.00012
    assert round(res[0, 0], 2) == 0.98
    print("test_softmax")


def test_normalize():
    x = fun.np.array([
        [0, 3, 4],
        [1, 6, 4]])
    res = fun.normalize(x)
    assert res[0, 1] == 0.6
    assert round(res[1, 1], 2) == 0.82
    assert res[0, 2] == 0.8
    print("test_normalize")
