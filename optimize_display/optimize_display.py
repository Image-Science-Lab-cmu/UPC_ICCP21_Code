import os
from os.path import join
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from util.visualizer import Visualizer

from wave_optics import Camera, set_params, imfilter, capture, wiener_deconv, get_wiener_loss, add_photon_noise
from loss import Loss
from utils import kron, print_opt


def load_data():
    # tf dataloader todo
    batch_size = 12
    # batch_size = 6
    img_height = 512
    img_width = 512

    train_ds = tf.data.Dataset.list_files('../../HQ/train/*.png', shuffle=False)
    train_ds = train_ds.shuffle(240, reshuffle_each_iteration=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def resize_and_rescale(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img)
        img = tf.cast(img, dtype=tf.float64) / 255

        # data augmentation
        img = tf.image.random_crop(img, size=tf.constant([img_height, img_width, 3]))
        return img

    train_ds = train_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    return train_ds


def optimize_pattern_with_data(opt, log_dir):

    vis = Visualizer(opt)

    tf.keras.backend.set_floatx('float64')
    # todo: need to figure out cameraOpt
    cameraOpt = set_params()
    camera = Camera(delta1=opt.delta1)
    optimizer = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=0.5)
    vars = []
    criterion = Loss(opt, cameraOpt)

    os.makedirs(log_dir, exist_ok=True)
    logfile = open('%s/log.txt'%log_dir, 'w')

    # print opt parameters
    print_opt(logfile, opt)

    # load init display pattern
    pattern = Image.open(opt.path).convert('L')
    pattern = (np.array(pattern) > 128).astype(np.float)
    # if opt.transparency > 0:
    #     pattern += opt.transparency
    #     pattern[pattern > 1]=1
    if opt.num_unit_mask > 1:
        pattern = np.repeat(pattern[:,:,None], opt.num_unit_mask, axis=2)

    if opt.amplitude_mask:
        pattern = tf.Variable(initial_value=(pattern * 2 - 1), trainable=True)
        vars.append(pattern)
    else:
        pattern = tf.Variable(initial_value=pattern, trainable=False)

    # aperture size
    sz = np.ceil(cameraOpt['D1'] / cameraOpt['delta1'] / opt.phase_resolution).astype(np.int)
    if opt.phase_mask:
        height_map = tf.Variable(initial_value=np.zeros((sz, sz)), trainable=True)
        vars.append(height_map)
    else:
        height_map = tf.zeros([sz,sz], dtype=tf.float64)

    # optimize random order
    if opt.random_order:
        np.random.seed(10)
        N =100  # todo set a large enough number
        num_patterns = 4 # todo:
        order_logits = np.random.rand(N, N, num_patterns)
        order_logits = tf.Variable(initial_value=order_logits, trainable=True)
        vars.append(order_logits)

    # load training dataset
    train_ds = load_data()

    losses = []
    avg_losses = []
    best_loss = 1e18
    epochs = 150
    # todo
    slope = 1
    for epoch_id in range(epochs):

        # slope = 1 + (3.5e-4 * batch_id) ** 2
        for batch_id, batch_img in enumerate(train_ds):
            with tf.GradientTape() as g:
                if opt.amplitude_mask:
                    mapped_pattern = tf.sigmoid(pattern* slope)
                else:
                    mapped_pattern = pattern

                if opt.random_order:
                    order = tf.nn.softmax(order_logits, axis=2)
                    order_ids = tf.argmax(order, axis=2)
                else:
                    order_ids = None

                kron_height_map = kron(height_map, tf.ones([opt.phase_resolution, opt.phase_resolution]))
                PSFs, u1_R, h1 = camera(mapped_pattern, height_map=kron_height_map,
                                        tile_option=opt.tile_option, random_order=order_ids)
                # PSFs = camera.get_psf_3smooth(mapped_pattern, height_map=kron_height_map,
                #                          tile_option=opt.tile_option, random_order=order_ids)

                # Note: during optimization, we don't add noise to captured.
                captured = capture(batch_img, PSFs)
                # noisy_captured = add_photon_noise(captured)
                deconved = wiener_deconv(captured, PSFs)
                # import ipdb; ipdb.set_trace()
                # compute losses
                loss, kk = criterion(mapped_pattern, PSFs, deconved, batch_img, epoch=epoch_id)
                losses.append(loss['total_loss'].numpy())
                avg_losses.append(np.mean(losses[-10:]))

            gradients = g.gradient(loss['total_loss'], vars)
            optimizer.apply_gradients(zip(gradients, vars))
            # import ipdb; ipdb.set_trace()

            # visualization
            if batch_id % 50 == 0:
                visuals = {}
                if opt.amplitude_mask and opt.num_unit_mask == 1:
                    visuals['pattern'] = (255*mapped_pattern[:,:,None]).numpy().astype(np.uint8)
                if opt.amplitude_mask and opt.num_unit_mask > 1:
                    for mask_id in range(opt.num_unit_mask):
                        visuals['pattern_%s'%mask_id] = (255 * mapped_pattern[:, :, mask_id][:,:,None]).numpy().astype(np.uint8)

                if opt.phase_mask:
                    # phase = tf.math.angle(u1_R)[:,:,None].numpy()
                    # phase[phase>=np.pi]=0
                    phase = h1[:,:,None].numpy()
                    phase = (phase - np.amin(phase)) / (np.amax(phase)-np.amin(phase))
                    visuals['phase'] = (255*phase).astype(np.uint8)
                if opt.random_order:
                    visuals['order'] = (255*order_ids.numpy()[:,:,None]/4).astype(np.uint8)

                PSFimg = tf.math.log(PSFs / tf.reduce_max(PSFs))
                PSFimg = (PSFimg - tf.reduce_min(PSFimg)) / (tf.reduce_max(PSFimg) - tf.reduce_min(PSFimg))
                visuals['PSFs'] = (255 * PSFimg).numpy().astype(np.uint8)

                captured_img = captured[0,...].numpy()
                deconved_img = deconved[0,...].numpy()
                captured_img = (captured_img - np.amin(captured_img)) / (np.amax(captured_img) - np.amin(captured_img))
                deconved_img = (deconved_img - np.amin(deconved_img)) / (np.amax(deconved_img) - np.amin(deconved_img))
                visuals['captured_0'] = (255 * captured_img).astype(np.uint8)
                visuals['deconved_0'] = (255 * deconved_img).astype(np.uint8)
                vis.display_current_results(visuals, epoch_id)

                # avgPSF = tf.reduce_mean(PSFs, axis=2)
                sz = tf.shape(PSFs).numpy()[0]
                vis.plot_current_curve(PSFs[int(sz / 2), :, :].numpy(), 'PSFs', display_id=10)
                vis.plot_current_curve(kk[int(sz/2), :, :].numpy(), 'Transfer function', display_id=15)

                vis.plot_current_curve(avg_losses, 'total loss', display_id=9)
                vis.print_current_loss(epoch_id, loss, logfile)

            if loss['total_loss'] < best_loss:
                best_loss = loss['total_loss']

        # save temporary results
        if epoch_id % 10 == 0:
            cv2.imwrite('%s/captured.png' % log_dir, (captured[0, ...] * 255).numpy().astype(np.uint8))
            cv2.imwrite('%s/deconved.png' % log_dir, (deconved[0, ...] * 255).numpy().astype(np.uint8))
            # sio.savemat('%s/height_map.mat' % log_dir, {'h1': h1.numpy()})
            # sio.savemat('%s/u1.mat' % log_dir, {'u1': u1_R.numpy()})
            sio.savemat('%s/PSFs.mat' % log_dir, {'PSFs': PSFs.numpy()})
            if opt.num_unit_mask == 1:
                cv2.imwrite('%s/display_pattern.png' % log_dir, (mapped_pattern * 255).numpy())
            else:
                sio.savemat('%s/display_pattern.mat' % log_dir, {'pattern': mapped_pattern.numpy()})
            if opt.random_order:
                sio.savemat('%s/optimized_order.mat' % log_dir, {'order': order_ids.numpy()})

    logfile.close()

    return mapped_pattern


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # optimization options
    parser.add_argument('--delta1', type=float, default=8e-6, help='change delta1 to simulate different dpis')
    parser.add_argument('--tile_option', type=str, default='repeat', help='the way to tile the pattern into display[repeat|rot90|randomRot]')
    parser.add_argument('--area_gamma', type=float, default=10, help='area loss weight')
    parser.add_argument('--center_area_gamma', type=float, default=0, help='center area loss weight')
    parser.add_argument('--power_line_gamma', type=float, default=0, help='power line loss weight')
    parser.add_argument('--l2_gamma', type=float, default=1, help='top-10 L2 loss weight')
    parser.add_argument('--inv_gamma', type=float, default=1, help='invertible loss weight')

    parser.add_argument('--lr', type=float, default=1, help='learning rate')

    parser.add_argument('--area', type=float, default=0.20, help='ratio between opening area to total area')
    parser.add_argument('--channel', type=int, default=-1, help='optimize for color channel [0, 1, 2, -1(average)]')

    parser.add_argument('--invertible', action='store_true', help='require PSFs invertible')
    parser.add_argument('--use_data', action='store_true', help='use data to compute top10-l2 loss')
    parser.add_argument('--keep_center_zero', action='store_true', help='keep the center square to be zeros')
    parser.add_argument('--keep_power_line', action='store_true', help='keep horizontal and vertical power lines to be zeros')

    parser.add_argument('--transparency', type=float, default=0, help='transparency of amplitude mask at blocked region.')
    parser.add_argument('--path', type=str, default='aperture_toled_single.png', help='initial display pattern')
    parser.add_argument('--amplitude_mask', action='store_true', help='optimize amplitude mask')
    parser.add_argument('--num_unit_mask', type=int, default=1, help='number of unit display patterns')
    parser.add_argument('--phase_mask', action='store_true', help='optimize phase mask')
    parser.add_argument('--random_order', action='store_true', help='optimize the random order')
    parser.add_argument('--phase_resolution', type=int, default='1', help='resolution of phase mask')
    parser.add_argument('--isTrain', action='store_true', help='train')

    parser.add_argument('--log_dir', type=str, default='log/', help='used in test, load optimized parameters')

    # display options
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_port', type=int, default=8999, help='visdom port of the web display')
    parser.add_argument('--display_env', type=str, default='main', help='visdom environment of the web display')
    parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                             help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--checkpoints_dir', type=str, default='logs/')

    opt = parser.parse_args()
    opt.no_html=False

    # log_dir = join('log', datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = join('log', opt.display_env)
    if opt.use_data:
        mapped_pattern, invKernels = optimize_pattern_with_data(opt, log_dir)
    else:
        mapped_pattern, invKernels = optimize_pattern(opt, log_dir)
