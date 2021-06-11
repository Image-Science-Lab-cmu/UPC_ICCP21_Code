import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from wave_optics import get_wiener_loss
from utils import gauss_2d


class Loss(keras.Model):
    def __init__(self, opt, cameraOpt):
        super(Loss, self).__init__()
        self.area = opt.area
        self.opt = opt

    def get_area_loss(self, mapped_pattern, epoch=0, target_area=None):

        if target_area is None:
            area_offset = 0
            if self.opt.keep_center_zero:
                # area_offset += 0.275
                area_offset += 0.51
            if self.opt.keep_power_line:
                area_offset += 0.045

            target_area = max(self.area, 1 - 0.05 * (epoch // 5) - area_offset) # since we always keep the center as zeros

        def get_one_area_loss(mapped_pattern, target_area):
            area_ratio = tf.reduce_sum(mapped_pattern) / tf.cast(tf.reduce_prod(tf.shape(mapped_pattern)), tf.float64)
            return (area_ratio - target_area) ** 2, area_ratio, target_area

        if self.opt.num_unit_mask == 1:
            area_loss, area_ratio, _ = get_one_area_loss(mapped_pattern, target_area)
            return area_loss, area_ratio, target_area
        else:
            avg_area_loss = 0
            avg_area_ratio = 0
            for mask_id in range(self.opt.num_unit_mask):
                area_loss, area_ratio, _ = get_one_area_loss(mapped_pattern[:,:,mask_id], target_area)
                avg_area_loss += area_loss
                avg_area_ratio += area_ratio
            # area_loss /= self.opt.num_unit_mask
            # area_ratio /= self.opt.num_unit_mask
            avg_area_loss /= self.opt.num_unit_mask
            avg_area_ratio /= self.opt.num_unit_mask
            return avg_area_loss, avg_area_ratio, target_area

    def call(self, mapped_pattern, PSFs, recons, gt=None, height_map=None, epoch=0):
        total_loss = 0
        out = {}

        # if self.opt.num_unit_mask == 1:
        #     area_loss, area, target_area = self.get_area_loss(mapped_pattern,epoch=epoch)
        # else:
        #     area_loss = 0
        #     for mask_id in range(self.opt.num_unit_mask):
        #         curr_area_loss, area, target_area = self.get_area_loss(mapped_pattern[:,:,mask_id], epoch=epoch)
        #         area_loss += curr_area_loss
        #     area_loss /= self.opt.num_unit_mask
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
            # === current paper result ===
            # invert_loss = get_wiener_loss(PSFs)
            # total_loss -= self.opt.inv_gamma * invert_loss
            # === Jan 4th ====
            # all loss uses + sign.
            # The invert_loss contains a minus sign for maximizing.
            invert_loss, kk = get_wiener_loss(PSFs)
            total_loss += self.opt.inv_gamma * invert_loss # @Jan 5 (minimize loss)
            out['invertible_loss'] = self.opt.inv_gamma * invert_loss

        if self.opt.keep_center_zero:
            # hard center constraint with box
            # center_pattern = mapped_pattern[5:16, 5:16]
            # todo: change the center square size
            center_pattern = mapped_pattern[3:18, 3:18]
            center_area_loss, _, _ = self.get_area_loss(center_pattern, target_area=0)

            # soft center constraint with Gaussian map
            # cost_map = gauss_2d(shape=(21, 21), sigma=3)
            # center_pattern = mapped_pattern * cost_map
            # center_area_loss, _, _ = self.get_area_loss(center_pattern, target_area=0)

            out['center_area_loss'] = self.opt.center_area_gamma * center_area_loss
            total_loss += out['center_area_loss']

        if self.opt.keep_power_line:
            # fix horizontal and vertical power line locations
            if self.opt.num_unit_mask == 1:
                power_line_map = np.zeros([21,21])
            else:
                power_line_map = np.zeros([21, 21, self.opt.num_unit_mask])
            power_line_map[:, 10] = 1
            power_line_map[10, :] = 1
            power_line_map = tf.constant(power_line_map, dtype=tf.float64)

            power_line_pattern = power_line_map * mapped_pattern
            power_line_loss, _, _ = self.get_area_loss(power_line_pattern, target_area=0)
            out['power_line_loss'] = self.opt.power_line_gamma * power_line_loss
            total_loss += out['power_line_loss']

        out['total_loss'] = total_loss

        return out, kk