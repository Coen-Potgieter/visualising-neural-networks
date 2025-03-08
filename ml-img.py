import numpy as np
import pygame as py
import sys
import PIL.Image
import random
import brain

WIDTH = 600
HEIGHT = WIDTH

WIN = py.display.set_mode((WIDTH, HEIGHT))


FPS = 60


def randomize_img(img):

    # pixel_data = list(img.getdata())
    # random.shuffle(pixel_data)
    # img.putdata(pixel_data)

    dim = img.size[0]

    rand_pixels = [random.randint(0, 150) if random.randint(
        0, 100) > 90 else 0 for i in range(dim*dim)]
    img.putdata(rand_pixels)

    return img


def get_img_info(img):

    pixel_data = list(img.getdata())
    pixel_array = np.array(pixel_data)
    return pixel_array.reshape(img.size[0], -1)


def loc_pixel_col(img):

    pixel_mat = get_img_info(img)

    loc = []
    col = []
    for y in range(img.size[0]):
        for x in range(img.size[1]):
            loc.append((x, y))
            col.append(pixel_mat[y, x])

    return loc, col


def new_w_b_pixels(img, struct):

    w_b_per_pixel = []
    for i in range(img.size[0] * img.size[1]):
        w, b = brain.init_w_b(struct)
        w_b_per_pixel.append([w, b])

    return w_b_per_pixel


def normalized_inpts(img):

    width, height = img.size[0], img.size[1]

    normalized_pixels = []
    for x in range(width):
        for y in range(height):
            normalized_pixels.append(
                ((x / (width/2) - 1), (y / (height/2) - 1))
            )

    return normalized_pixels


def disp_img(img_mat, surf):

    w, h = img_mat.shape[0], img_mat.shape[1]

    pixel_w = WIDTH // w
    pixel_h = HEIGHT // h

    surf_pixels = py.surfarray.array3d(surf)

    for r in range(h):
        for c in range(w):
            col = img_mat[r][c]
            # print(c*pixel_w, (c+1)*pixel_w)
            surf_pixels[c*pixel_w:(c+1)*pixel_w, r *
                        pixel_h:(r+1)*pixel_h] = (col, col, col)

    py.surfarray.blit_array(surf, surf_pixels)


def ml_img(img, Y, pixel_w_b):

    lr = 0.1

    width, height = img.size[0], img.size[1]

    normalized_pixels = normalized_inpts(img)

    Y = [np.array([elem/255]).reshape((1, 1)) for elem in Y]

    pixels_1d = []
    for i, elem in enumerate(normalized_pixels):
        w = pixel_w_b[i][0]
        b = pixel_w_b[i][1]
        X = np.array([elem[0], elem[1]]).reshape((2, 1))

        # forward prop
        z, a = brain.for_prop(X, w, b)

        # back prop
        dw, db = brain.back_prop(X, z, a, w, Y[i])

        # use results from back prop to update w and biases
        w, b = brain.update_params(w, b, dw, db, lr)

        # gonna then use the results from first for prop
        pixels_1d.append(a[0][0][0]*255)

        pixel_w_b[i][0] = w
        pixel_w_b[i][1] = b

    return np.array(pixels_1d).reshape(width, height), pixel_w_b


def main():
    # ok mister ginger spice, my idea is this,
    # have 28x28 little ml brains each training, very small brain what like 2 weights and 1 bias per brain?
    # 28x28x3 = 2352 params

    # ok how to display then?
    # once again pygame is looking atractive
    # yeppers

    imgs = {
        1: "drawn",
        2: "flower_small",
        3: "chicago_skyline",
        4: "mountain",
        5: "Woman",
        6: "salad"
    }

    clock = py.time.Clock()
    img = PIL.Image.open(f"Assets/Pics/{imgs[3]}.png")

    canvas = py.Surface((WIDTH, HEIGHT))
    struct = [2, 1]

    # we need 28x28 of these guys

    # ight so a list of 28x28 elements
    # each element is a list of 2 elements containing w, b
    # each element in w and b being the layer
    # each have a numpy array where row is the node and col is the prev nodes weight

    # print(len(w_b_per_pixel))  # pixels
    # print(len(w_b_per_pixel[0]))    # w/b
    # print(len(w_b_per_pixel[0][0]))  # Layers
    # print(w_b_per_pixel[0][0][0].shape)  # actual weights of pixel1, w, layer1

    loc, col = loc_pixel_col(img)

    # w, b = brain.train_every_pixel(loc, col, w, b)

    # img = randomize_img(img)
    # pixel_mat = get_img_info(img)

    w_b_per_pixel = new_w_b_pixels(img, struct)
    ml_pixel_mat, w_b_per_pixel = ml_img(img, col, w_b_per_pixel)

    py.display.update()

    while 1:
        # img = randomize_img(img)
        # pixel_mat = get_img_info(img)

        # w_b_per_pixel = new_w_b_pixels(img, struct)

        ml_pixel_mat, w_b_per_pixel = ml_img(img, col, w_b_per_pixel)

        disp_img(ml_pixel_mat, canvas)

        WIN.blit(canvas, (0, 0))

        py.display.update()
        clock.tick(FPS)

        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                sys.exit()
    


if __name__ == "__main__":
    main()
