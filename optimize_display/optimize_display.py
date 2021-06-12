import os
from os.path import join
import numpy as np
import scipy.io as sio
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from util.visualizer import Visualizer

from wave_optics import Camera, set_params, capture, wiener_deconv, get_wiener_loss
from loss import Loss
from utils import print_opt


def load_data():
    # tf dataloader
    batch_size = 12
    img_height = 512
    img_width = 512

    # todo: change path to your training image directory
    train_ds = tf.data.Dataset.list_files('Train/Toled/HQ/*.png', shuffle=False)
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


def optimize_pattern_with_data(opt):

    tf.keras.backend.set_floatx('float64')

    # visualization and log
    vis = Visualizer(opt)
    log_dir = join('log', opt.display_env)  # directory that saves optimized pattern and training log
    os.makedirs(log_dir, exist_ok=True)
    logfile = open('%s/log.txt' % log_dir, 'w')
    print_opt(logfile, opt)  # print opt parameters

    # set up camera
    cameraOpt = set_params()
    camera = Camera()

    # set up optimization
    optimizer = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=0.5)
    criterion = Loss(opt, cameraOpt)
    vars = []               # variables which we track gradient
    train_ds = load_data()  # load training dataset

    # initial pixel opening as all-open
    pattern = np.ones((21, 21), dtype=np.float)
    pattern = tf.Variable(initial_value=(pattern * 2 - 1), trainable=True)  # later we use sigmoid to map 0 to 1.
    vars.append(pattern)

    losses = []         # loss for each batch
    avg_losses = []     # smoothed losses
    best_loss = 1e18
    epochs = 150
    for epoch_id in range(epochs):

        for batch_id, batch_img in enumerate(train_ds):
            with tf.GradientTape() as g:
                # map pattern values to range [0, 1]
                mapped_pattern = tf.sigmoid(pattern)
                # compute PSF from pixel opening pattern and tiling method
                PSFs = camera(mapped_pattern, tile_option=opt.tile_option)

                # apply PSF to images
                captured = capture(batch_img, PSFs)
                deconved = wiener_deconv(captured, PSFs)

                # compute losses
                loss, transfer_funcs = criterion(mapped_pattern, PSFs, deconved, batch_img, epoch=epoch_id)
                losses.append(loss['total_loss'].numpy())
                avg_losses.append(np.mean(losses[-10:]))  # average losses from the latest 10 epochs

            gradients = g.gradient(loss['total_loss'], vars)
            optimizer.apply_gradients(zip(gradients, vars))

            # visualization and log
            if batch_id % 50 == 0:
                # visualize images
                visuals = {}
                visuals['pattern'] = (255*mapped_pattern[:,:,None]).numpy().astype(np.uint8)  # pixel opening pattern
                visuals['PSFs'] = vis.tensor_to_img(tf.math.log(PSFs / tf.reduce_max(PSFs)))  # PSF in log-scale
                visuals['captured_0'] = vis.tensor_to_img(captured)  # captured image in current batch
                visuals['deconved_0'] = vis.tensor_to_img(deconved)  # deblurred image in current batch
                vis.display_current_results(visuals, epoch_id)

                # plot curves
                sz = tf.shape(PSFs).numpy()[0]
                vis.plot_current_curve(PSFs[int(sz / 2), :, :].numpy(), 'PSFs', display_id=10)  # a slice of PSF (ideally a Dirac delta function)
                vis.plot_current_curve(transfer_funcs[int(sz/2), :, :].numpy(), 'Transfer function', display_id=15)
                                                                                 # a slice of transfer functions (ideally all-ones)
                vis.plot_current_curve(avg_losses, 'Total loss', display_id=9)   # losses

                # print losses to log file
                vis.print_current_loss(epoch_id, loss, logfile)

            if loss['total_loss'] < best_loss:
                best_loss = loss['total_loss']

        # save temporary results
        if epoch_id % 10 == 0:
            sio.savemat('%s/PSFs.mat' % log_dir, {'PSFs': PSFs.numpy()})
            cv2.imwrite('%s/display_pattern.png' % log_dir, (mapped_pattern * 255).numpy())

    logfile.close()

    return mapped_pattern


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # optimization options
    parser.add_argument('--tile_option', type=str, default='repeat', help='pixel tiling methods [repeat|randomRot]')
    # parser.add_argument('--use_data', action='store_true', help='use data-driven loss top-10 L2')
    # parser.add_argument('--invertible', action='store_true', help='use PSF-induced loss L_inv')

    parser.add_argument('--area', type=float, default=0.20, help='target pixel opening ratio 0~1')
    parser.add_argument('--area_gamma', type=float, default=10, help='area constraint weight')
    parser.add_argument('--l2_gamma', type=float, default=10, help='top-10 L2 loss weight')
    parser.add_argument('--inv_gamma', type=float, default=0.01, help='L_inv loss weight')

    parser.add_argument('--log_dir', type=str, default='log/', help='save optimized pattern and training log')
    parser.add_argument('--isTrain', action='store_true', help='train')
    parser.add_argument('--lr', type=float, default=1, help='learning rate')

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
    opt.no_html = False
    opt.isTrain = True
    opt.use_data = True
    opt.invertible = True

    mapped_pattern = optimize_pattern_with_data(opt)

    # python optimize_display.py --tile_option repeat --area_gamma 10 --l2_gamma 10 --inv_gamma 0.01 --display_env VIS_NAME
