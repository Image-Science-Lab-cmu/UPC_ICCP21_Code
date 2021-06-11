import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift

from utils import load_display, load_rect




class LensProp(layers.Layer):
    def __init__(self):
        super(LensProp, self).__init__()

    def call(self, u1, dx1, dx2, lambd, f, d):
        m, _ = tf.shape(u1).numpy()
        k = 2 * np.pi / lambd

        L2 = m * dx2
        x2 = tf.constant(np.arange(-m / 2, m / 2) * dx2)
        X2, Y2 = tf.meshgrid(x2, x2)

        j = tf.complex(real=np.zeros_like(X2),
                       imag=np.ones_like(X2))

        c_imag = k * (1 - d / f) / (2 * f) * (tf.pow(X2, 2) + tf.pow(Y2, 2))
        c_real = np.zeros_like(c_imag)
        c0 = tf.exp(tf.complex(real=c_real, imag=c_imag))
        c = tf.multiply(-j / (lambd * f), c0)

        u2_1 = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(u1))) * dx1 ** 2
        u2 = tf.multiply(c, u2_1)
        return u2, L2


def capture(img, psf):
    '''
    convolve images with psf
    :param img_batch:  tensor [n, h, w ,c]
    :param psf:        tensor [hp, wp, c]
    :return:           tensor [n, h, w, c]
    '''
    n, h, w, c = tf.shape(img).numpy()
    hp, wp, _ = tf.shape(psf).numpy()
    img = tf.cast(img, dtype=tf.float64)
    psf = tf.cast(psf, dtype=tf.float64)

    img_blur = []
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_real = tf.pad(psf[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        img_complex = tf.complex(real=img_real, imag=tf.zeros_like(img_real))
        psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

        fft_img = tf.signal.fft2d(img_complex)
        fft_psf = tf.signal.fft2d(psf_complex)
        fft_out = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1]))

        out = tf.abs(tf.signal.ifft2d(fft_out))
        crop_x, crop_y = int(hp/2), int(wp/2)
        out = out[:, crop_x: crop_x + h, crop_y: crop_y + w]
        img_blur.append(out)

    return tf.stack(img_blur, axis=3)


def wiener_deconv(img, psf):
    '''
    convolve images with psf
    :param img_batch:  tensor [n, h, w ,c]
    :param psf:        tensor [hp, wp, c]
    :return:           tensor [n, h, w, c]
    '''
    n, h, w, c = tf.shape(img).numpy()
    hp, wp, _ = tf.shape(psf).numpy()
    img = tf.cast(img, dtype=tf.float64)
    psf = tf.cast(psf, dtype=tf.float64)

    img_recon = []
    snr = 0.0001 / tf.math.reduce_std(img) ** 2
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        psf_real = tf.pad(psf[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        img_complex = tf.complex(real=img_real, imag=tf.zeros_like(img_real))
        psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

        fft_img = tf.signal.fft2d(img_complex)
        fft_psf = tf.signal.fft2d(psf_complex)

        # wiener deconvolution kernel
        invK = tf.complex(real=1 / (tf.abs(fft_psf) ** 2 + snr), imag=tf.zeros_like(psf_real))
        K = tf.math.conj(fft_psf) * invK
        fft_out = fft_img * tf.tile(K[None, :, :], tf.constant([n, 1, 1]))

        out = tf.abs(tf.signal.ifftshift(tf.signal.ifft2d(fft_out), axes=(1, 2)))
        crop_x0, crop_y0 = int(h/2), int(w/2)

        if crop_x0 + h > h + hp:
            offset = int(h/2 - hp)
            out = tf.roll(out, shift=[-offset, -offset], axis=[1,2])
            # import ipdb; ipdb.set_trace()
            crop_x0 -= offset
            crop_y0 -= offset
        out = out[:, crop_x0:crop_x0+h, crop_y0: crop_y0+w]
        img_recon.append(out)

    return tf.stack(img_recon, axis=3)


def get_wiener_loss(psf):
    '''
    This function computes the invertible loss.
    :param psf:
    :return: scalar
    '''
    hp, wp, _ = tf.shape(psf).numpy()
    psf = tf.cast(psf, dtype=tf.float64)

    def get_overall_func(blur_func):

        blur_func = blur_func / tf.reduce_sum(blur_func)  # normalize to one
        blur_func = tf.complex(real=blur_func, imag=tf.zeros_like(blur_func))

        fft_blur_func = tf.signal.fft2d(blur_func)
        inv_fft_blur_func = tf.complex(real=1 / (tf.abs(fft_blur_func) ** 2 + 0.015),
                                       imag=tf.zeros([hp, wp], dtype=tf.float64))
        overall_func = tf.abs(tf.math.conj(fft_blur_func) * fft_blur_func * inv_fft_blur_func)
        return tf.signal.fftshift(overall_func)

    # compute system frequency response for RGB channels
    overall_funcs = []
    for channel in range(3):
        overall_funcs.append(get_overall_func(psf[:,:,channel]))
    overall_funcs = tf.stack(overall_funcs, axis=2)

    # compute invertibility loss
    sorted_overall_funcs = tf.sort(tf.reshape(overall_funcs, [-1]), direction='ASCENDING')
    num_low = int(0.3 * hp * wp * 3)
    score = -tf.reduce_mean(sorted_overall_funcs[:num_low])

    return score, overall_funcs


class Camera(keras.Model):
    def __init__(self, delta1=8e-6):
        super(Camera, self).__init__()
        self.lens2sensor = LensProp()

        self.D1 = 4e-3
        self.wvls = {'R': 0.61e-6,
                     'G': 0.53e-6,
                     'B': 0.47e-6}
        self.f = 0.01  # focal length [m]
        self.delta1 = delta1  # spacing of lens aperture [m]
        self.delta2 = 0.5e-6  # spacing of sensor aperture [m]
        self.pixel_sz = 2e-6  # pixel pitch size [m]

        self.M_R = self.get_nsamples(self.wvls['R'])
        self.M_G = self.get_nsamples(self.wvls['G'])
        self.M_B = self.get_nsamples(self.wvls['B'])

        # generate aperture mask of max size
        self.aperture_R = tf.constant(load_rect(self.delta1, self.D1, self.M_R))

    def get_nsamples(self, wvl):
        return int((wvl * self.f
                    / self.delta1 / self.delta2) // 2 * 2)  # number of samples

    def dn_sample(self, psf):
        dn_scale = int(self.pixel_sz / self.delta2)
        kernel = tf.constant(np.ones((dn_scale, dn_scale, 1, 1)), dtype=tf.float64)
        return tf.squeeze(tf.nn.conv2d(psf[None, :, :, None], kernel, strides=dn_scale, padding='VALID'))

    def call(self, pattern, tile_option=None):

        T = tf.shape(pattern)[0].numpy()
        display = load_display(pattern, self.delta1, self.M_R, self.delta1 * T, tile_option)
        u1_real = display * self.aperture_R

        u1 = tf.complex(real=u1_real, imag=tf.zeros_like(u1_real))

        crop_G = int((self.M_R - self.M_G) / 2)
        crop_B = int((self.M_R - self.M_B) / 2)

        # image_red
        u1_R = u1

        # image_green
        u1_G = u1[crop_G:-crop_G, crop_G:-crop_G]

        # image_blue
        u1_B = u1[crop_B:-crop_B, crop_B:-crop_B]

        # lens propagation
        u2_R, _ = self.lens2sensor(u1_R, self.delta1, self.delta2, self.wvls['R'], self.f, 0)
        u2_G, _ = self.lens2sensor(u1_G, self.delta1, self.delta2, self.wvls['G'], self.f, 0)
        u2_B, _ = self.lens2sensor(u1_B, self.delta1, self.delta2, self.wvls['B'], self.f, 0)

        dn_u2_R = self.dn_sample(tf.pow(tf.abs(u2_R), 2))
        dn_u2_G = self.dn_sample(tf.pow(tf.abs(u2_G), 2))
        dn_u2_B = self.dn_sample(tf.pow(tf.abs(u2_B), 2))

        # cat three channels
        crop_G = int((dn_u2_G.shape[0] - dn_u2_B.shape[0]) / 2)
        crop_R = int((dn_u2_R.shape[0] - dn_u2_B.shape[0]) / 2)

        PSFs = tf.stack([dn_u2_R[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
                         dn_u2_G[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
                         dn_u2_B], axis=2)

        return PSFs


def set_params():
    args = dict()
    args['D1'] = 4e-3  # diam of lens/disp aperture [m]
    args['f'] = 0.01  # focal length [m]
    args['z'] = 0.05  # display lens distance [m]

    args['delta1'] = 8e-6  # spacing of lens aperture [m]
    args['delta2'] = 0.5e-6  # spacing of sensor aperture [m]
    args['pixel_sz'] = 2e-6  # pixel pitch size [m]

    args['wvls'] = {'R': 0.61e-6,
                    'G': 0.53e-6,
                    'B': 0.47e-6}

    def get_nsamples(wvl):
        m = np.ceil(wvl * args['f']
                    / args['delta1'] / args['delta2'])  # number of samples
        return int(m // 2 * 2)

    args['M_R'] = get_nsamples(args['wvls']['R'])
    args['M_G'] = get_nsamples(args['wvls']['G'])
    args['M_B'] = get_nsamples(args['wvls']['B'])
    return args