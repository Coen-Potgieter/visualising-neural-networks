from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import brain
import math

BLACK = "#000000"

# style 1
BLUE = "#0A91AB"
ORANGE = "#E25E3E"

# style 2
BEIGE = "#F5EDCE"
LIGHT_BLUE = "#B4CDE6"

# style 3
GREY = "#EFEFEF"
YELLOW = "#FFDE00"

RED = "#FF0000"

styles = [(BLUE, ORANGE),
          (BEIGE, LIGHT_BLUE),
          (GREY, YELLOW)]
# ---------------------- Appearance ----------------- #
style_choice = 0

LINE_COL, DOT_COL = styles[style_choice][0], styles[style_choice][1]
BG = BLACK

# ---------------------- MLP performance ----------------- #
LR = 0.0004

struct_choice = {
    0: [1, 10, 5, 3, 1],    # simple
    1: [1, 100, 75, 50, 25, 1]  # Complex
}
STRUCT = struct_choice[1]

# ---------------------- Function to approximate ----------------- #
SAMPLE_SIZE = 200
FUNC_IDX = 3
# 0: Straight Line
# 1: Parabola
# 2: Sin
# 3: weird shit
# 4: Sigmoid
# 5: ReLu
# 6: Softplus
# 7: abs_sinh
# 8: sum_of_sins
# 9: sum_of_polys
# 10: cant exactly explain this bro

# something to is that this thing is rlly, rlly good at plotting the sigmoid function,
# even if it gets given 2 points it chooses to go through both in a sigmoid shape,
# this is becuase the output activation is a sigmoid, which is cool :)


def all_funcs(x, idx):

    def line(x):
        return (x-25)*2 + 10

    def sin(x):
        return math.sin(x*0.3) + 10

    def parabola(x):
        return 0.1*(x-25)**2 + 10

    def my_random_thing(x):
        return abs((1/(0.1 * abs(x-25)+0.1)) + 1)

    def sigmoid(x):
        return 10 / (1 + math.exp(-0.3 * (x - 25))) + 10

    def ReLu(x):
        return max(25, x)

    def softplus(x):
        return (10 * (x-35)) / (8 + 0.02 * math.exp(-(x-35))) + 10

    def abs_sinh(x):

        b1 = 0.1
        b2 = 0.5
        s = 25
        return min(abs(b1 * math.sinh(b2*(x-s)) + 10), 20)

    def sum_of_sins(x):

        a = 4.3
        n = 0.1
        s = 52
        return a * abs(2 * math.sin(n*3*(x-s)) + math.cos(n*2*(x-s)) + 0.5 * math.sin(n*5*(x-s)) + 0.25 * math.cos(n*7*(x-s)))

    def sum_of_polys(x):
        A = 0.05
        a1 = 0.13
        a2 = -2.4
        a3 = -47.7
        s = 18

        x = x-s
        return A * abs(a1*pow(x, 3) + a2*(x**2) + a3*x) + 10

    def bro_dont_ask(x):
        a0 = 0.66
        a1 = -5.9
        a2 = -17.3
        a3 = 0.78
        a4 = -6.6
        p1 = -2.8
        p2 = 0.24
        s = 27

        return min(abs(a0*abs(a1*pow(x-s+0.001, p1) + a2*pow(x-s, p2) + a4*math.sin(a3*(x-s))) - 13), 10)
    
    funcs = [line, parabola, sin, my_random_thing,
             sigmoid, ReLu, softplus, abs_sinh, sum_of_sins, sum_of_polys, bro_dont_ask]
    return funcs[idx](x)
    # just something to take note is that all ive really implented is x^n, varoius sinusoidal things
    # and piecewise operations, i can still do exponential, ln, even like uneven x shifts, ie.
    # x+a x+b, cuase right now all x's are shifted by 25.
    # no need to really do this tho considering can be see that no matter how complex the function is,
    # the MLP can plot it. (bro dont ask is most complicated i think), complex values?!?!


def linspace(start, end, num=100):
    step = (end - start) / num

    x = [start]
    while len(x) < num:
        x.append(x[-1] + step)

    return x


def rand_func(x):

    y = []
    x_to_the = random.randint(1, 3)
    x_coeff = random.randint(0, 10)
    c = random.randint(-10, 10)

    for elem in x:
        y.append(pow(elem, x_to_the) * x_coeff + c)

    return y


def unnormalize(x, minx, maxx):
    return [(elem+2) * ((maxx-minx)/4) + minx for elem in x]


def normailze(x):
    min_x = min(x)
    max_x = max(x) - min_x

    return [((elem-min_x) / (max_x/4)) - 2 for elem in x]

    pass


def ml_line(domain, w, b, max_y, min_y):
    y = []
    for elem in domain:
        _, a = brain.for_prop(elem, w, b, act_func=5, out_func=1)
        y.append(unnormalize(a[-1][0] * 4 - 2, min_y, max_y))

    return y


def ml_train(X, Y, w, b):
    global LR
    examples = len(Y[0])

    running_sum = 0

    def grad_calc():
        nonlocal running_sum
        running_sum += np.sum(np.abs(dw))

    for i in range(examples):
        # for prop
        z, a = brain.for_prop(X[0, i], w, b, act_func=5, out_func=1)
        # back prop
        dw, db = brain.back_prop_func(X[0, i], z, a, w, Y[0, i], act_func=5)

        # grad_calc()

        # update w, b
        w, b = brain.update_params(w, b, dw, db, LR)

    # print(running_sum)

    return w, b


def main():

    # right, one input(x), and one output(y), but need more complexity than just 1 to 1 with wieght
    # and bias becuase then its actually just a linear thing? a1*w1 + b = y
    # so need hidden layers now its y = prev(a1)*w1 + prev(a2)*w2 ..... + b
    # additionally these activations run through activation functions like Leaky ReLu for instance
    # so this can be very complex?

    w, b = brain.init_w_b(STRUCT)
    x = [random.randint(0, 50) for i in range(SAMPLE_SIZE)]
    min_x = min(x)
    max_x = max(x)
    un_normal_domain = linspace(min_x, max_x, num=500)

    y = [all_funcs(elem, FUNC_IDX) for elem in x]
    y_max = max(y)
    y_min = min(y)

    # inpts and outputs never change so might aswell normalize and proccess here
    # gonna normalize, Y outputs wont be normailized

    # i think im gonna normalize x to between -2 and 2 cuase thay works nice with hyptan()?
    # ended up using leaky ReLu, so this is fucking stupid
    X = normailze(x)
    domain = linspace(min(X), max(X), num=len(un_normal_domain))
    X = np.array(X).reshape(1, -1)

    # shit aint working so normalizing y vals aswell
    Y = [(elem+2)/4 for elem in normailze(y)]
    Y = np.array(Y).reshape(1, -1)

    def anim_rand(i):
        nonlocal line_approx
        line_approx.remove()

        ml_y = rand_func(domain)
        line_approx, = axes.plot(un_normal_domain, ml_y)
        axes.set_xbound(min_x - 5, max_x+5)
        axes.set_ybound(y_min-5, y_max+5)

    def anim_ml(i):
        nonlocal line_approx, w, b
        line_approx.remove()
        w, b = ml_train(X, Y, w, b)
        ml_y = ml_line(domain, w, b, y_max, y_min)

        # idk if thing fucks out then retry
        if np.isnan(ml_y[0]):
            print("happened")
            w, b = brain.init_w_b(STRUCT)

        line_approx, = axes.plot(
            un_normal_domain, ml_y, color=LINE_COL, linewidth=3.5)
        axes.set_xbound(min_x - 5, max_x+5)
        axes.set_ybound(y_min-5, y_max+5)

    fig, axes = plt.subplots(1, 1)

    # ------------ figure config ----------- #
    fig.set_size_inches(w=12, h=6)
    fig.set_facecolor(BG)
    axes.set_axis_off()
    axes.set_xbound(min_x - 5, max_x+5)
    axes.set_ybound(y_min-5, y_max+5)

    # plot dots
    axes.scatter(x, y, color=DOT_COL)

    # init line
    line_approx, = axes.plot(0, 1)

    # diff animations
    ani2 = FuncAnimation(plt.gcf(), anim_ml, interval=1)
    # ani1 = FuncAnimation(plt.gcf(), anim_rand, interval=500)

    plt.show()


if __name__ == "__main__":
    main()
