# Optimize display pattern of under-display
# camera by optimizing
# input variable using Tensorflow.
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import random

from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift

from utils import load_display, load_rect

import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


# tf.debugging.set_log_device_placement(True)

class LensProp(layers.Layer):
    def __init__(self):
        super(LensProp, self).__init__()

    def call(self, u1, dx1, dx2, lambd, f, d):
        m, _ = tf.shape(u1).numpy()
        #         dx1 = delta1
        k = 2 * np.pi / lambd

        #         L2 = lambd*f/dx1
        #         dx2 = lambd*f/L1
        #         x2 = tf.constant(np.arange(-L2/2, L2/2, dx2))
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
    # img_blur_full = []
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        # psf_real = tf.pad(psf[:,:,cc]/tf.reduce_sum(psf[:,:,cc]), tf.constant([[0,h], [0,w]]), 'CONSTANT')
        psf_real = tf.pad(psf[:, :, cc], tf.constant([[0, h], [0, w]]), 'CONSTANT')

        img_complex = tf.complex(real=img_real, imag=tf.zeros_like(img_real))
        psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

        fft_img = tf.signal.fft2d(img_complex)
        fft_psf = tf.signal.fft2d(psf_complex)
        fft_out = fft_img * tf.tile(fft_psf[None, :, :], tf.constant([n, 1, 1]))

        out = tf.abs(tf.signal.ifft2d(fft_out))
        # img_blur_full.append(out)
        crop_x, crop_y = int(hp/2), int(wp/2)
        # out = out[:, 146:658, 146:658]
        out = out[:, crop_x: crop_x + h, crop_y: crop_y + w]
        img_blur.append(out)

    # import scipy.io as sio
    # img_blur_full3 = tf.stack(img_blur_full, axis=3)
    # sio.savemat('test_img_blur.mat', {'img_blur_full': img_blur_full3.numpy(),
    #                                   'img_gt': img.numpy()})
    # import ipdb; ipdb.set_trace()
    return tf.stack(img_blur, axis=3)
    # todo
    # return tf.stack(img_blur, axis=3) / tf.reduce_max(img_blur)


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
    # img_blur_full = []
    snr = 0.0001 / tf.math.reduce_std(img) ** 2
    for cc in range(c):
        img_real = tf.pad(img[:, :, :, cc], tf.constant([[0, 0], [0, hp], [0, wp]]), 'CONSTANT')
        # psf_real = tf.pad(psf[:,:,cc]/tf.reduce_sum(psf[:,:,cc]), tf.constant([[0,h], [0,w]]), 'CONSTANT')
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
        # img_blur_full.append(out)
        crop_x0, crop_y0 = int(h/2), int(w/2)
        # out = out[:, 256:768, 256:768]

        if crop_x0 + h > h + hp:
            offset = int(h/2 - hp)
            out = tf.roll(out, shift=[-offset, -offset], axis=[1,2])
            # import ipdb; ipdb.set_trace()
            crop_x0 -= offset
            crop_y0 -= offset
        out = out[:, crop_x0:crop_x0+h, crop_y0: crop_y0+w]
        # out = tf.roll(out, shift=[-100, -100], axis=[1, 2])
        # out = out[:, hp:hp+h, wp:wp+w]
        img_recon.append(out)

    # # todo
    # import scipy.io as sio
    # img_blur_full3 = tf.stack(img_blur_full, axis=3)
    # sio.savemat('test_img_recons.mat', {'img_recons': img_blur_full3.numpy()})

    return tf.stack(img_recon, axis=3)


def add_photon_noise(x):


    # sensor = {'capacity': 15506,
    #           'noise_std': 7.34,
    #           # 'gain': 1 / 5637
    #           'gain': 1 / 1608}
    # sensor info
    capacity = 15506
    Ls = [273, 654, 1608, 4005, 10024]
    time = 1

    # normalize images to max intensity 1
    # import ipdb; ipdb.set_trace()
    max_intensities = tf.reduce_max(x, [1,2,3], keepdims=True)
    # x = x / max_intensities

    L = random.choice(Ls)
    x_photons = x / max_intensities * L * time
    std = tf.stop_gradient(tf.sqrt(x_photons))
    x_photons_noisy = x_photons + std * tf.random.normal(x.shape, dtype=tf.float64)
    x_noisy_clamp = tf.minimum(x_photons_noisy, capacity * tf.ones(x.shape, dtype=tf.float64))
    # ipdb.set_trace()
    return x_noisy_clamp / L * max_intensities


def get_wiener_loss_dep(psf):
    '''
    deprecated function:
    This function is used to max/min (1-K)^2,
    which is not consisten with paper.
    This is used in top-10L2 + inv random.
    compute the K .* K' of psf
    :param psf:
    :return: scalar
    '''
    h, w = 1024, 2048
    hp, wp, _ = tf.shape(psf).numpy()
    psf = tf.cast(psf, dtype=tf.float64)
    psf = (psf[:, :, 0] / tf.reduce_sum(psf[:, :, 0]) +
           psf[:, :, 1] / tf.reduce_sum(psf[:, :, 1]) +
           psf[:, :, 2] / tf.reduce_sum(psf[:, :, 2])) / 3

    psf_real = tf.pad(psf[:, :], tf.constant([[0, h], [0, w]]), 'CONSTANT')
    psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

    # fourier transform of kernel
    fft_psf = tf.signal.fft2d(psf_complex)

    # wiener deconvolution kernel
    snr = 0.015
    invK = tf.complex(real=1 / (tf.abs(fft_psf) ** 2 + snr), imag=tf.zeros_like(psf_real))
    K = tf.math.conj(fft_psf) * invK

    # psf * wienerK
    # take the average of low-10 of KK
    nh, nw = tf.shape(K).numpy()
    fft_psf_K = tf.reshape(tf.abs(fft_psf * K), [-1])
    # fft_psf_K = tf.sort(fft_psf_K, direction='ASCENDING')
    fft_psf_K = tf.sort(tf.pow(1-fft_psf_K, 2), direction='DESCENDING')
    num_low = int(0.3 * nh * nw)
    return tf.reduce_mean(fft_psf_K[:num_low])


def get_wiener_loss_original(psf):
    '''
    This is the version that consistent with paper
    that is used to maximize the invertible loss.
    top10 L2 + inv /  repeat used this version.
    top10 L2 + inv / random has similar results as
    the deprecated version. Performance difference
    mainly comes from variance.
    compute the K .* K' of psf
    :param psf:
    :return: scalar
    '''
    h, w = 1024, 2048
    hp, wp, _ = tf.shape(psf).numpy()
    psf = tf.cast(psf, dtype=tf.float64)
    psf = (psf[:, :, 0] / tf.reduce_sum(psf[:, :, 0]) +
           psf[:, :, 1] / tf.reduce_sum(psf[:, :, 1]) +
           psf[:, :, 2] / tf.reduce_sum(psf[:, :, 2])) / 3

    psf_real = tf.pad(psf[:, :], tf.constant([[0, h], [0, w]]), 'CONSTANT')
    psf_complex = tf.complex(real=psf_real, imag=tf.zeros_like(psf_real))

    # fourier transform of kernel
    fft_psf = tf.signal.fft2d(psf_complex)

    # wiener deconvolution kernel
    snr = 0.015
    invK = tf.complex(real=1 / (tf.abs(fft_psf) ** 2 + snr), imag=tf.zeros_like(psf_real))
    K = tf.math.conj(fft_psf) * invK

    # psf * wienerK
    # take the average of low-30 of KK
    nh, nw = tf.shape(K).numpy()
    fft_psf_K = tf.reshape(tf.abs(fft_psf * K), [-1])
    fft_psf_K = tf.sort(fft_psf_K, direction='ASCENDING')
    num_low = int(0.3 * nh * nw)

    # todo: visualize system transfer function
    kk = tf.abs(fft_psf * K)
    kk = tf.signal.fftshift(kk)
    kk = kk[int(nh/2), :]

    return -tf.reduce_mean(fft_psf_K[:num_low]), kk


def get_wiener_loss(psf):
    '''
    This is the version that consistent with paper
    that is used to maximize the invertible score.
    top10 L2 + inv /  repeat used this version.
    top10 L2 + inv / random has similar results as
    the deprecated version. Performance difference
    mainly comes from variance.
    compute the K .* K' of psf
    :param psf:
    :return: scalar
    '''
    # import ipdb;
    hp, wp, _ = tf.shape(psf).numpy()
    psf = tf.cast(psf, dtype=tf.float64)

    def get_overall_func(blur_func):

        blur_func = blur_func / tf.reduce_sum(blur_func)  # normalize to one
        blur_func = tf.complex(real=blur_func, imag=tf.zeros_like(blur_func))

        fft_blur_func = tf.signal.fft2d(blur_func)
        inv_fft_blur_func = tf.complex(real=1 / (tf.abs(fft_blur_func) ** 2 + 0.015),
                                       imag=tf.zeros([hp, wp], dtype=tf.float64))
        overall_func = tf.abs(tf.math.conj(fft_blur_func) * fft_blur_func * inv_fft_blur_func)
        # ipdb.set_trace()
        return tf.signal.fftshift(overall_func)

    # compute system functions for RGB channels
    overall_funcs = []
    for channel in range(3):
        overall_funcs.append(get_overall_func(psf[:,:,channel]))
    overall_funcs = tf.stack(overall_funcs, axis=2)

    # compute invertibility loss
    sorted_overall_funcs = tf.sort(tf.reshape(overall_funcs, [-1]), direction='ASCENDING')
    num_low = int(0.3 * hp * wp * 3)
    score = -tf.reduce_mean(sorted_overall_funcs[:num_low])

    # # todo: visualize system transfer function
    # kk = tf.abs(fft_psf * K)
    # kk = tf.signal.fftshift(kk)
    # kk = kk[int(nh/2), :]

    # ipdb.set_trace()
    return score, overall_funcs


class FresnelProp(layers.Layer):
    def __init__(self):
        super(FresnelProp, self).__init__()

    def call(self, u1, L, lambd, z):
        m, n = tf.shape(u1).numpy()
        dx = L / m
        k = 2 * np.pi / lambd

        fx = tf.constant(np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / L))
        FX, FY = tf.meshgrid(fx, fx)

        H_imag = -np.pi * lambd * z * (tf.pow(FX, 2) + tf.pow(FY, 2))
        H_real = np.zeros_like(H_imag)
        H = tf.exp(tf.complex(real=H_real, imag=H_imag))
        H = tf.signal.fftshift(H)

        U1 = tf.signal.fft2d(tf.signal.fftshift(u1))
        U2 = tf.multiply(H, U1)
        u2 = tf.signal.ifftshift(tf.signal.ifft2d(U2))
        return u2, L


# todo:
class FresnelASP(layers.Layer):
    def __init__(self):
        super(FresnelASP, self).__init__()

    def ft2(self, u, delta):
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(u))) * delta ** 2

    def ift2(self, u, delta_f):
        m, _ = tf.shape(u).numpy()
        return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(u))) * (m * delta_f) ** 2

    def call(self, u1, lambd, delta1, delta2, z):
        n, _ = tf.shape(u1).numpy()
        k = 2 * np.pi / lambd

        # source plane coordinates
        x1 = tf.constant(np.arange(-n / 2, n / 2, 1) * delta1)
        XX1, YY1 = tf.meshgrid(x1, x1)
        r1sq = tf.pow(XX1, 2) + tf.pow(YY1, 2)

        delta_f1 = 1 / (n * delta1)
        fx1 = tf.constant(np.arange(-n / 2, n / 2, 1) * delta_f1)
        FX1, FY1 = tf.meshgrid(fx1, fx1)
        fsq = tf.pow(FX1, 2) + tf.pow(FY1, 2)

        m = delta2 / delta1
        x2 = tf.constant(np.arange(-n / 2, n / 2, 1) * delta2)
        XX2, YY2 = tf.meshgrid(x2, x2)
        r2sq = tf.pow(XX2, 2) + tf.pow(YY2, 2)

        # quadratic phase factors
        Q1_var_imag = k / 2 * (1-m) / z * r1sq
        Q2_var_imag = -np.pi**2 * 2 * z / m / k * fsq
        Q3_var_imag = k / 2 * (m-1) / (m*z) * r2sq

        Q1 = tf.exp(tf.complex(real=np.zeros_like(Q1_var_imag), imag=Q1_var_imag))
        Q2 = tf.exp(tf.complex(real=np.zeros_like(Q2_var_imag), imag=Q2_var_imag))
        Q3 = tf.exp(tf.complex(real=np.zeros_like(Q3_var_imag), imag=Q3_var_imag))

        # compute the propagated field
        out = Q3 * self.ift2(Q2 * self.ft2(Q1 * u1 / m, delta1), delta_f1)
        return out


def phaseModulation(wvl, height_map, coeff):
    k = 2 * np.pi / wvl
    Phi_imag = k * (coeff - 1) * height_map
    Phi_real = np.zeros_like(Phi_imag)
    out = tf.exp(tf.complex(real=Phi_real, imag=Phi_imag))
    return out


def padtensor(tensor, siz):
    hsiz = tf.shape(tensor)
    pad = tf.cast(tf.math.ceil((siz - hsiz) / 2), dtype=tf.uint64).numpy()
    paddings = tf.constant([[pad[0], pad[0]], [pad[1], pad[1]]])
    pad_height_map = tf.pad(tensor, paddings)
    return pad_height_map[:siz[0], :siz[1]]


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
        # psf_transpose = tf.transpose(psf, perm=[2,0,1])[:,:,:,None]
        # dn_psf = tf.nn.conv2d(psf_transpose, kernel, strides=dn_scale, padding='VALID')
        # return tf.transpose(tf.squeeze(dn_psf), perm=[1,2,0])
        return tf.squeeze(tf.nn.conv2d(psf[None, :, :, None], kernel, strides=dn_scale, padding='VALID'))
        # return tf.squeeze(tf.nn.avg_pool(psf[None,:,:,:], dn_scale, dn_scale, 'VALID', name='dn_psf'))

    def get_one_expected_psf(self, b1, b2, b3, b4, deltas, M, R):

        fb1 = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(b1[:M, :M])))
        fb2 = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(b2[:M, :M])))
        fb3 = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(b3[:M, :M])))
        fb4 = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(b4[:M, :M])))
        fdeltas = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(deltas[:M, :M])))

        return R*R *(tf.pow(tf.abs(fb1), 2) + tf.pow(tf.abs(fb2), 2) + tf.pow(tf.abs(fb3), 2)) + \
                tf.pow(tf.abs(fb4), 2) * tf.pow(tf.abs(fdeltas), 2)

    # @ Mar 1 Anqi
    def get_expected_psf(self, pattern):

        T = tf.shape(pattern)[0].numpy()
        R = np.round(self.D1 / self.delta1 / T)

        # create four versions of pattern
        m1 = pattern
        m2 = tf.transpose(pattern)
        m3 = pattern[:, ::-1]
        m4 = m2[::-1, :]

        # create averaged versions
        paddings = tf.constant([[0, self.M_R - T], [0, self.M_R - T]])
        b1 = tf.complex(real=tf.pad(0.25*(m3-m4-m1+m2), paddings, 'CONSTANT'), imag=tf.zeros([self.M_R, self.M_R], tf.float64))
        b2 = tf.complex(real=tf.pad(0.25*(m3-m4+m1-m2), paddings, 'CONSTANT'), imag=tf.zeros([self.M_R, self.M_R], tf.float64))
        b3 = tf.complex(real=tf.pad(0.25*(m3+m4-m1-m2), paddings, 'CONSTANT'), imag=tf.zeros([self.M_R, self.M_R], tf.float64))
        b4 = tf.complex(real=tf.pad(0.25*(m3+m4+m1+m2), paddings, 'CONSTANT'), imag=tf.zeros([self.M_R, self.M_R], tf.float64))

        # create 2D delta train
        delta = np.zeros((21, 21)); delta[11, 11] = 1
        delta = tf.constant(delta, tf.float64)
        deltas = tf.complex(real=self.aperture_R*load_display(delta, self.delta1, self.M_R, self.delta1 * T, None, None),
                            imag=tf.zeros([self.M_R, self.M_R], tf.float64))

        # compute their fourier transforms for three wavelengths
        I2_R = self.get_one_expected_psf(b1, b2, b3, b4, deltas, self.M_R, R)
        I2_G = self.get_one_expected_psf(b1, b2, b3, b4, deltas, self.M_G, R)
        I2_B = self.get_one_expected_psf(b1, b2, b3, b4, deltas, self.M_B, R)

        # downsample to sensor pitch
        dn_u2_R = self.dn_sample(I2_R)
        dn_u2_G = self.dn_sample(I2_G)
        dn_u2_B = self.dn_sample(I2_B)

        # cat three channels
        crop_G = int((dn_u2_G.shape[0] - dn_u2_B.shape[0]) / 2)
        crop_R = int((dn_u2_R.shape[0] - dn_u2_B.shape[0]) / 2)

        PSFs = tf.stack([dn_u2_R[crop_R:crop_R + dn_u2_B.shape[0], crop_R:crop_R + dn_u2_B.shape[0]],
                         dn_u2_G[crop_G:crop_G + dn_u2_B.shape[0], crop_G:crop_G + dn_u2_B.shape[0]],
                         dn_u2_B], axis=2)

        return PSFs


    def call(self, pattern, height_map=None, tile_option=None, random_order=None):

        T = tf.shape(pattern)[0].numpy()
        display = load_display(pattern, self.delta1, self.M_R, self.delta1 * T, tile_option, random_order)
        u1_real = display * self.aperture_R



        u1 = tf.complex(real=u1_real, imag=tf.zeros_like(u1_real))
        if height_map is not None:
            # todo
            # import scipy.io as sio
            # h1 = sio.loadmat('height_map.mat')['h1']
            h1 = padtensor(height_map, tf.shape(u1))  # pad height map

        crop_G = int((self.M_R - self.M_G) / 2)
        crop_B = int((self.M_R - self.M_B) / 2)

        # image_red
        u1_R = u1
        if height_map is not None:
            h1_R = h1
            u1_R *= phaseModulation(self.wvls['R'], h1_R, 1.4)

        # image_green
        u1_G = u1[crop_G:-crop_G, crop_G:-crop_G]
        if height_map is not None:
            h1_G = h1[crop_G:-crop_G, crop_G:-crop_G]
            u1_G *= phaseModulation(self.wvls['G'], h1_G, 1.4)

        # image_blue
        u1_B = u1[crop_B:-crop_B, crop_B:-crop_B]
        if height_map is not None:
            h1_B = h1[crop_B:-crop_B, crop_B:-crop_B]
            u1_B *= phaseModulation(self.wvls['B'], h1_B, 1.4)

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

        return PSFs, u1, h1
        # return PSFs, u1_R

    def get_psf_3smooth(self, pattern, height_map=None, tile_option=None, random_order=None):

        # densely sampled wavelengths
        wvls_R = np.arange(-0.025e-6, 0.025e-6, 0.01e-6) + 0.61e-6
        wvls_G = np.arange(-0.025e-6, 0.025e-6, 0.01e-6) + 0.53e-6
        wvls_B = np.arange(-0.025e-6, 0.025e-6, 0.01e-6) + 0.47e-6

        # todo: debug
        # wvls_R = np.arange(-0.0e-6, 0.025e-6, 0.025e-6) + 0.61e-6
        # wvls_G = np.arange(-0.0e-6, 0.025e-6, 0.025e-6) + 0.53e-6
        # wvls_B = np.arange(-0.0e-6, 0.025e-6, 0.025e-6) + 0.47e-6
        N = len(wvls_R)

        # number of samples for each wavelengths
        M_Rs = [self.get_nsamples(wvl) for wvl in wvls_R]
        M_Gs = [self.get_nsamples(wvl) for wvl in wvls_G]
        M_Bs = [self.get_nsamples(wvl) for wvl in wvls_B]

        M_max = max(M_Rs)
        M_min = min(M_Bs)

        u2_sz = int(M_min / (self.pixel_sz / self.delta2))

        # initialize input plane (max number of samples)
        T = tf.shape(pattern)[0].numpy()
        display = load_display(pattern, self.delta1, M_max, self.delta1 * T, tile_option, random_order)
        aperture = tf.constant(load_rect(self.delta1, self.D1, M_max))
        u1_real = display * aperture

        u1 = tf.complex(real=u1_real, imag=tf.zeros_like(u1_real))

        crop_Rs = [int((M_max - M) / 2) for M in M_Rs]
        crop_Gs = [int((M_max - M) / 2) for M in M_Gs]
        crop_Bs = [int((M_max - M) / 2) for M in M_Bs]

        # # image_red
        # u1_R = u1
        #
        # # image_green
        # u1_G = u1[crop_G:-crop_G, crop_G:-crop_G]
        #
        # # image_blue
        # u1_B = u1[crop_B:-crop_B, crop_B:-crop_B]
        #
        # # lens propagation
        # u2_R, _ = self.lens2sensor(u1_R, self.delta1, self.delta2, self.wvls['R'], self.f, 0)
        # u2_G, _ = self.lens2sensor(u1_G, self.delta1, self.delta2, self.wvls['G'], self.f, 0)
        # u2_B, _ = self.lens2sensor(u1_B, self.delta1, self.delta2, self.wvls['B'], self.f, 0)
        #
        # dn_u2_R = self.dn_sample(tf.pow(tf.abs(u2_R), 2))
        # dn_u2_G = self.dn_sample(tf.pow(tf.abs(u2_G), 2))
        # dn_u2_B = self.dn_sample(tf.pow(tf.abs(u2_B), 2))
        #
        # # cat three channels
        # crop_G = int((dn_u2_G.shape[0] - dn_u2_B.shape[0]) / 2)
        # crop_R = int((dn_u2_R.shape[0] - dn_u2_B.shape[0]) / 2)

        def get_psf_single_wavelength(u1, wvl, out_sz):
            u2, _ = self.lens2sensor(u1, self.delta1, self.delta2, wvl, self.f, 0)
            dn_u2 = self.dn_sample(tf.pow(tf.abs(u2), 2))
            crop = int((dn_u2.shape[0] - out_sz) / 2)
            return dn_u2[crop:crop + out_sz, crop:crop + out_sz]

        # PSF red channel
        u2_buffer = []
        for ii in range(N):
            u2_buffer.append(get_psf_single_wavelength(u1[crop_Rs[ii]:M_Rs[ii]-crop_Rs[ii], crop_Rs[ii]:M_Rs[ii]-crop_Rs[ii]], wvls_R[ii], u2_sz))
        u2_R = tf.reduce_mean(tf.stack(u2_buffer, axis=2), axis=2)

        # PSF green channel
        u2_buffer = []
        for ii in range(N):
            u2_buffer.append(get_psf_single_wavelength(u1[crop_Gs[ii]:-crop_Gs[ii], crop_Gs[ii]:-crop_Gs[ii]], wvls_G[ii], u2_sz))
        u2_G = tf.reduce_mean(tf.stack(u2_buffer, axis=2), axis=2)

        # PSF blue channel
        u2_buffer = []
        for ii in range(N):
            u2_buffer.append(get_psf_single_wavelength(u1[crop_Bs[ii]:-crop_Bs[ii], crop_Bs[ii]:-crop_Bs[ii]], wvls_B[ii], u2_sz))
        u2_B = tf.reduce_mean(tf.stack(u2_buffer, axis=2), axis=2)

        PSFs = tf.stack([u2_R,
                         u2_G,
                         u2_B], axis=2)

        # import ipdb; ipdb.set_trace()
        return PSFs
        # return PSFs, u1_R


def set_params():
    args = dict()
    args['D1'] = 4e-3  # diam of lens/disp aperture [m]
    #         args['lambda'] =lambd                  # wavelength
    #         args['wvls'] = {'R':0.61e-6,
    #                          'G':0.53e-6,
    #                          'B':0.47e-6}
    #         args['k'] =2*np.pi/args['lambda']      # wavenumber
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


def imfilter(kernels, image):
    '''
    apply kernel to image
    :param kernels: (tensor) [kN, kM, 3]  float value 0~1
    :param image:   (tensor) [iN, iM, 3]  float value 0~255
    :return: out:   (tensor) [iN, iM, 3]  float value 0~255
    '''

    Isize = tf.shape(image).numpy()
    ksize = tf.shape(kernels).numpy()
    siz = Isize + ksize - 1

    # pad image
    pad_image = tf.constant([[0, siz[0] - Isize[0]], [0, siz[1] - Isize[1]], [0, 0]])
    image = tf.pad(image, pad_image, 'CONSTANT')
    image = tf.complex(real=image, imag=tf.zeros_like(image))
    # pad kernels
    pad_kernel = tf.constant([[0, siz[0] - ksize[0]], [0, siz[1] - ksize[1]], [0, 0]])
    kernels = tf.pad(kernels, pad_kernel, 'CONSTANT')
    kernels = tf.complex(real=kernels, imag=tf.zeros_like(kernels))

    # init filtered image
    out = []
    for cc in range(3):
        kernel = kernels[:, :, cc]
        kernel = kernel / tf.reduce_sum(kernel)
        Fx = fft2d(image[:, :, cc])
        Fk = fft2d(kernel)
        out.append(tf.math.real(ifft2d(Fx * Fk)))
    out = tf.stack(out, axis=2)
    pad = ((siz - Isize) / 2).astype(int)
    return out[pad[0]:-pad[0], pad[1]:-pad[1], :]
