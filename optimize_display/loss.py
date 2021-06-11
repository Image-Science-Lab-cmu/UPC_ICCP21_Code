import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from wave_optics import get_wiener_loss


class Loss(keras.Model):
    def __init__(self, opt, cameraOpt):
        super(Loss, self).__init__()
        self.area = opt.area
        self.opt = opt

    def get_area_loss(self, mapped_pattern, epoch=0, target_area=None):

        if target_area is None:
            target_area = max(self.area, 1 - 0.05 * (epoch // 5))  # decrease target area every 5 epochs

        area_ratio = tf.reduce_sum(mapped_pattern) / tf.cast(tf.reduce_prod(tf.shape(mapped_pattern)), tf.float64)
        area_loss = (area_ratio - target_area) ** 2
        return area_loss, area_ratio, target_area

    def call(self, mapped_pattern, PSFs, recons, gt=None, epoch=0):
        total_loss = 0
        out = {}

        area_loss, area, target_area = self.get_area_loss(mapped_pattern, epoch=epoch)
        total_loss += self.opt.area_gamma * area_loss
        out['area'] = area
        out['target_area'] = tf.constant(target_area)
        out['area_loss'] = self.opt.area_gamma * area_loss

        # top-10 l2 loss
        if self.opt.use_data:
            n, w, h, c = tf.shape(gt).numpy()
            residual = tf.abs(tf.reshape(gt - recons, [n, -1]))
            residual = tf.sort(residual, axis=1, direction='DESCENDING')
            num_top = int(0.1 * w * h * c)
            residual = residual[:, :num_top]
            out['top_l2'] = self.opt.l2_gamma * tf.nn.l2_loss(residual) / num_top
            total_loss += out['top_l2']

        if self.opt.invertible:
            invert_loss, transfer_funcs = get_wiener_loss(PSFs)
            total_loss += self.opt.inv_gamma * invert_loss  # (minimize loss, invertible loss has minus sign inside)
            out['invertible_loss'] = self.opt.inv_gamma * invert_loss

        out['total_loss'] = total_loss

        return out, transfer_funcs