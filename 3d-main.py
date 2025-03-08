from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import brain

RED = "#FF0000"

BLACK = "#000000"

BG = BLACK

SAMPLE_SIZE = 200
XRANGE = (0, 50)
YRANGE = (0, 50)

TURN_SPEED = 0.75


good_cmaps = {
    2: 'Blues', 3: 'Blues_r', 10: 'CMRmap', 16: 'Greens',
    18: 'Greys', 19: 'Greys_r', 20: 'OrRd', 21: 'OrRd_r', 22: 'Oranges',
    23: 'Oranges_r', 37: 'PuBu_r', 54: 'Reds', 55: 'Reds_r', 64: 'Wistia',
    65: 'Wistia_r', 74: 'afmhot', 75: 'afmhot_r', 76: 'autumn',
    77: 'autumn_r', 78: 'binary', 79: 'binary_r', 80: 'bone', 81: 'bone_r',
    86: 'cividis', 87: 'cividis_r', 92: 'copper', 93: 'copper_r',
    116: 'gray', 117: 'gray_r', 122: 'inferno', 123: 'inferno_r',
    126: 'magma', 127: 'magma_r', 134: 'plasma', 135: 'plasma_r',
    144: 'summer', 145: 'summer_r', 156: 'turbo', 157: 'turbo_r',
    158: 'twilight', 159: 'twilight_r', 160: 'twilight_shifted',
    161: 'twilight_shifted_r', 162: 'viridis', 163: 'viridis_r',
    164: 'winter', 165: 'winter_r'}

CMAP = good_cmaps[162]

FUNC_IDX = 0
STRUCT = [2, 10, 5, 1]
LR = 0.01

def all_funcs(x, y, idx):

    def func1(x, y):
        return pow(x-25, 2) + pow(y-25, 2)

    def func2(x, y):
        return pow(x-25, 2) - pow(y-25, 2)

    def cone(x, y):
        a = 1
        b = 1
        c = 1
        return np.sqrt(pow(c, 2) * ((pow(x-25, 2) / pow(a, 2)) + (pow(y-25, 2) / pow(b, 2)) + 1))

    def circle(x, y):
        r = 20
        return random.choice([-1, 1]) * np.sqrt(pow(r, 2) - pow(y-25, 2) - pow(x-25, 2)) + 25

    funcs = [func1, func2, cone, circle]
    return funcs[idx](x, y)

def linspace(start, end, num=100):
    step = (end - start) / num

    x = [start]
    while len(x) < num:
        x.append(x[-1] + step)

    return x


def normailze(x):
    min_x = min(x)
    max_x = max(x) - min_x

    return [((elem-min_x) / (max_x/4)) - 2 for elem in x]


def unnormalize(x, minx, maxx):
    return [(elem+2) * ((maxx)/4) + minx for elem in x]


def ml_line(domain, w, b, max_z, min_z):

    num_inpts = domain.shape[1]

    z = np.zeros((num_inpts, num_inpts), dtype=float)

    xs = domain[0, :]
    ys = domain[1, :]

    for x_idx in range(num_inpts):
        for y_idx in range(num_inpts):
            elem = np.array([xs[x_idx], ys[y_idx]]).reshape(2, 1)

            _, a = brain.for_prop(elem, w, b, act_func=5, out_func=1)

            z[y_idx, x_idx] = unnormalize(
                a[-1][0] * 4 - 2, min_z, max_z)[0]

    return z


def ml_train(X, Y, w, b):

    num_inpts = Y.shape[1]

    for i in range(num_inpts):
        inpt = X[:, i].reshape(2, 1)
        z, a = brain.for_prop(inpt, w, b, act_func=5, out_func=1)
        dw, db = brain.back_prop_func(inpt, z, a, w, Y[:, i], act_func=5)
        w, b = brain.update_params(w, b, dw, db, LR)

    return w, b


def main():

    # ------------------- data prep --------------- #
    x = [random.randint(XRANGE[0], XRANGE[1]) for i in range(SAMPLE_SIZE)]
    y = [random.randint(YRANGE[0], YRANGE[1]) for i in range(SAMPLE_SIZE)]
    z = [all_funcs(x[i], y[i], FUNC_IDX) for i in range(len(x))]

    # for unnormalize
    max_z = max(z)
    min_z = min(z)

    # for plotting
    un_normal_x = linspace(min(x), max(x), num=25)
    un_normal_y = linspace(min(y), max(y), num=len(un_normal_x))
    mshgrd_x, mshgrd_y = np.meshgrid(un_normal_x, un_normal_y)

    # for inpts to MLP
    normal_x = normailze(x)
    normal_y = normailze(y)
    X_x = np.array(normal_x).reshape(1, -1)
    X_y = np.array(normal_y).reshape(1, -1)
    X = np.append(X_x, X_y, axis=0)

    # for forward prop
    inpt_domain_x = linspace(min(normal_x), max(
        normal_x), num=len(un_normal_x))
    inpt_domain_y = linspace(min(normal_y), max(
        normal_y), num=len(un_normal_x))

    domain_X_x = np.array(inpt_domain_x).reshape(1, -1)
    domain_X_y = np.array(inpt_domain_y).reshape(1, -1)

    inpt_domain = np.append(domain_X_x, domain_X_y, axis=0)

    # normalize outputs aswell
    ph = normailze(z)

    # sgimoid
    normal_z = [(elem + 2) / 4 for elem in ph]
    Y = np.array(normal_z).reshape(1, -1)
    # --------------- ml stuffs init --------------- #

    w, b = brain.init_w_b(STRUCT)

    w, b = ml_train(X, Y, w, b)
    mshgrd_z = ml_line(inpt_domain, w, b, max_z, min_z)

    def anim_ml(i):
        nonlocal surf_approx, w, b

        surf_approx.remove()
        mshgrd_z = ml_line(inpt_domain, w, b, max_z, min_z)

        if np.isnan(mshgrd_z).any():
            w, b = brain.init_w_b(STRUCT)
            print("Reset Weighs and biases")

        surf_approx = axes.plot_surface(
            mshgrd_x, mshgrd_y, mshgrd_z, cmap=CMAP, alpha=1)

    # ------------------ animation -------------- #
    turn_angle = 30

    def turn_anim(i):
        nonlocal turn_angle, w, b
        axes.view_init(elev=30, azim=turn_angle, roll=0)
        w, b = ml_train(X, Y, w, b)

        turn_angle += TURN_SPEED

    # ------------ figure config ----------- #
    fig = plt.figure()
    fig.set_size_inches(w=12, h=6)
    fig.set_facecolor(BLACK)

    # ------------ axes config ----------- #
    axes = fig.add_subplot(111, projection='3d')

    axes.set_facecolor(BLACK)
    axes.xaxis.pane.set_facecolor(BLACK)
    axes.yaxis.pane.set_facecolor(BLACK)
    axes.zaxis.pane.set_facecolor(BLACK)

    axes.xaxis._axinfo["grid"]['color'] = BLACK
    axes.yaxis._axinfo["grid"]['color'] = BLACK
    axes.zaxis._axinfo["grid"]['color'] = BLACK

    axes.xaxis.pane.set_edgecolor(BLACK)
    axes.yaxis.pane.set_edgecolor(BLACK)
    axes.zaxis.pane.set_edgecolor(BLACK)

    # ----------------------------------------------- #

    axes.scatter(x, y, z, c="#FF0000", marker="o", s=20, alpha=1)

    surf_approx = axes.plot_surface(
        mshgrd_x, mshgrd_y, mshgrd_z, cmap=CMAP)

    ani2 = FuncAnimation(plt.gcf(), turn_anim, interval=1)
    ani1 = FuncAnimation(plt.gcf(), anim_ml, interval=50)

    plt.show()

    pass


if __name__ == "__main__":
    main()
