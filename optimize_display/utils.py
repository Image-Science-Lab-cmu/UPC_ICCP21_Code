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


# def load_aperture(path, total_width, display_width, im_width, total_resolution):

#     # init input tensor
#     im = Image.open(path).convert('L')

#     # resize image
#     im_resolution = int(im_width/total_width*total_resolution)
#     im = im.resize((im_resolution, im_resolution))
#     im = (np.array(im) > 128).astype(np.float)

#     # repeat image to form display
#     n_repeat = int(display_width / im_width)
#     display = np.tile(im, (n_repeat, n_repeat))

#     # resize display
#     pad = int((total_resolution - im_resolution*n_repeat + 1)/2)
#     in_field_real = np.pad(display, ((pad, pad), (pad, pad)))
#     in_field_real = in_field_real[:total_resolution, :total_resolution]
#     return display

# def load_aperture(pattern, L1, M, s):
def load_display(pattern, delta, M, s, option, random_order=None):
    # repeat display patterns to cover aperture plane.
    # pattern: display pattern (tensor.float)
    # L1: diam of the source plane [m]
    #     (display and padding region)
    # D1: diam of aperture [m]
    # M:  number of samples
    # s:  diam of pattern [m]
    # ds: spacing of pattern [m]

    if option == 'rot90':
        # rotate and flip the pattern
        rot90 = tf.transpose(pattern)
        flip = pattern[:, ::-1]
        rot90_flip = rot90[::-1, :]
        pattern4 = tf.concat([tf.concat([pattern, rot90], axis=1),
                              tf.concat([rot90_flip, flip], axis=1)], axis=0)
        c = tf.constant([delta * M / (2 * s), delta * M / (2 * s)])
        c = tf.dtypes.cast(tf.math.ceil(c), tf.int32)
        display = tf.tile(pattern4, c)

    elif option == 'randomRot' and random_order is None:
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

    elif option == 'randomRot' and random_order is not None:
        # local randomness
        # rotate and flip the pattern
        rot90 = tf.transpose(pattern)
        flip = pattern[:, ::-1]
        rot90_flip = rot90[::-1, :]
        patterns = tf.stack([pattern, rot90, flip, rot90_flip], axis=0)


        N = int(np.ceil(delta * M / s))
        # pattern16 = []
        # for ix in range(N):
        #     pattern4 = []
        #     for iy in range(N):
        #         # pattern4.append(patterns[:,:,random_order[ix, iy]])
        #         pattern4.append(tf.squeeze(tf.gather_nd(patterns, [[random_order[ix,iy]]])))
        #     pattern16.append(tf.concat(pattern4, axis=1))
        # display = tf.concat(pattern16, axis=0)
        out = tf.gather_nd(patterns, random_order[:N, :N, None])
        display = tf.reshape(out, [N*21, N*21])
        import ipdb; ipdb.set_trace()

    elif option == 'randomPatterns':
        num_pattern = tf.shape(pattern)[2]
        N = int(np.ceil(delta * M / s))
        np.random.seed(10)
        order = np.random.randint(num_pattern, size=(N, N))
        pattern16 = []
        for ix in range(N):
            pattern4 = []
            for iy in range(N):
                pattern4.append(pattern[:,:,order[ix,iy]])
            pattern16.append(tf.concat(pattern4, axis=1))
        display = tf.concat(pattern16, axis=0)
        # import ipdb; ipdb.set_trace()
    else:
        # figure out times to repeat the pattern
        c = tf.constant([delta * M / s, delta * M / s])
        c = tf.dtypes.cast(tf.math.ceil(c), tf.int32)
        display = tf.tile(pattern, c)

    # crop display size to the size of aperture plane
    return tf.cast(display[:M, :M], dtype=tf.float64)


def random_rot_and_flip(pattern):
    val = np.random.rand()
    if val < 0.25:
        return pattern
    elif val < 0.5:
        return tf.transpose(pattern)
    elif val < 0.75:
        return tf.transpose(pattern)[::-1, :]
    else:
        return pattern[:, ::-1]


def gauss_2d(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # sumh = h.sum()
    # if sumh != 0:
    #     h /= sumh
    return tf.constant(h)


def kron(a, b):
    i, k, s = tf.shape(a)[0], tf.shape(b)[0], tf.shape(b)[0]
    o = s * (i - 1) + k

    a_tf = tf.reshape(tf.cast(a, dtype=tf.float64), [1, i, i, 1])
    b_tf = tf.reshape(tf.cast(b, dtype=tf.float64), [k, k, 1, 1])

    return tf.squeeze(tf.nn.conv2d_transpose(a_tf, b_tf, (1, o, o, 1), [1, s, s, 1], "VALID"))


def print_opt(file, opt):
    print('=========================================')
    file.write('=========================================\n')
    for arg in vars(opt):
        print('%s: %s' % (arg, getattr(opt, arg)))
        file.write('%s: %s\n' % (arg, getattr(opt, arg)))
    print('=========================================')
    file.write('=========================================\n')