#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

num_predict_per_level=len(cfg.ANCHOR.ANCHOR_RATIOS)


class SSDHead():




    def __call__(self,fms,L2_reg,training=True,ratios_per_pixel=num_predict_per_level):

        cla_set=[]
        reg_set=[]
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('ssdout'):

                for i in range(len(fms)):
                    current_feature = fms[i]

                    dim_h=tf.shape(current_feature)[1]
                    dim_w = tf.shape(current_feature)[2]
                    #feature_halo=halo(current_feature,'fm%d'%i)
                    feature_halo=current_feature
                    reg_out = slim.conv2d(feature_halo, ratios_per_pixel*4, [1, 1], stride=1, activation_fn=None,normalizer_fn=None, scope='out_reg%d'%i)
                    cls_out = slim.conv2d(feature_halo, ratios_per_pixel*cfg.DATA.num_class, [1, 1], stride=1, activation_fn=None,normalizer_fn=None, scope='out_cla%d'%i)

                    reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w * ratios_per_pixel, 4]))
                    cls_out = tf.reshape(cls_out, ([-1, dim_h * dim_w* ratios_per_pixel,cfg.DATA.num_class]))


                    cla_set.append(cls_out)
                    reg_set.append(reg_out)

                reg = tf.concat(reg_set, axis=1)
                cla = tf.concat(cla_set, axis=1)

        return reg,cla





class SSDHeadSingle():




    def __call__(self,fms,L2_reg,training=True,ratios_per_pixel=num_predict_per_level):

        cla_set=[]
        reg_set=[]
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('ssdout'):

                for i in range(len(fms)):
                    current_feature = fms[i]

                    dim_h=tf.shape(current_feature)[1]
                    dim_w = tf.shape(current_feature)[2]
                    #feature_halo=halo(current_feature,'fm%d'%i)
                    feature_halo=current_feature
                    reg_out = slim.conv2d(feature_halo, ratios_per_pixel*4, [1, 1], stride=1, activation_fn=None,normalizer_fn=None, scope='out_reg%d'%i)

                    cla_out = slim.conv2d(feature_halo, ratios_per_pixel, [1, 1], stride=1, activation_fn=None,normalizer_fn=None, scope='out_cla%d'%i)

                    reg_out = tf.reshape(reg_out, ([-1, dim_h, dim_w, ratios_per_pixel, 4]))
                    reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w * ratios_per_pixel, 4]))

                    cla_out = tf.reshape(cla_out, ([-1, dim_h, dim_w, ratios_per_pixel, 1]))
                    cla_out = tf.reshape(cla_out, ([-1, dim_h * dim_w* ratios_per_pixel,1]))


                    cla_set.append(cla_out)
                    reg_set.append(reg_out)



                reg = tf.concat(reg_set, axis=1)
                cla = tf.concat(cla_set, axis=1)
        return reg,cla