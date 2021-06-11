import numpy as np
import tensorflow as tf


####################### load inital display pattern ##########################

def load_rect(dx, display_width, total_resolution):
    x1 = np.arange(-total_resolution / 2, total_resolution / 2) * dx
    X1, Y1 = np.meshgrid(x1, x1)
    in_field_real = rect(X1 / display_width) * rect(Y1 / display_width)
    return in_field_real


def rect(x):
    return (np.abs(x) <= 0.5).astype(np.float)


def load_display(pattern, delta, M, s, option):
    # repeat display patterns to cover aperture plane.
    # pattern: display pattern (tensor.float)
    # L1: diam of the source plane [m]
    #     (display and padding region)
    # D1: diam of aperture [m]
    # M:  number of samples
    # s:  diam of pattern [m]
    # ds: spacing of pattern [m]

    if option == 'randomRot':
        # local randomness
        # rotate and flip the pattern
        rot90 = tf.transpose(pattern)
        flip = pattern[:, ::-1]
        rot90_flip = rot90[::-1, :]

        N = int(np.ceil(delta * M / s))
        np.random.seed(10)
        order = np.random.randint(4, size=(N, N))

        pattern16 = []
        for ix in range(N):
            pattern4 = []
            for iy in range(N):
                if order[ix, iy] == 0:
                    pattern4.append(pattern)
                elif order[ix, iy] == 1:
                    pattern4.append(rot90)
                elif order[ix, iy] == 2:
                    pattern4.append(flip)
                else:
                    pattern4.append(rot90_flip)
            pattern16.append(tf.concat(pattern4, axis=1))
        display = tf.concat(pattern16, axis=0)

    elif option == 'repeat':
        # figure out times to repeat the pattern
        c = tf.constant([delta * M / s, delta * M / s])
        c = tf.dtypes.cast(tf.math.ceil(c), tf.int32)
        display = tf.tile(pattern, c)

    else:
        print('Invalid pixel tiling method.')

    # crop display size to the size of aperture plane
    return tf.cast(display[:M, :M], dtype=tf.float64)


def print_opt(file, opt):
    print('=========================================')
    file.write('=========================================\n')
    for arg in vars(opt):
        print('%s: %s' % (arg, getattr(opt, arg)))
        file.write('%s: %s\n' % (arg, getattr(opt, arg)))
    print('=========================================')
    file.write('=========================================\n')