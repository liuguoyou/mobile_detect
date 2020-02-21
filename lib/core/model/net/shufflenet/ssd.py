#-*-coding:utf-8-*-


import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode,batch_decode_fix

from lib.core.model.net.shufflenet.shufflenet import shufflenet_v2,shufflenet_v2_fpn

from lib.core.model.loss.ssd_loss import ssd_loss
from train_config import config as cfg

from lib.helper.logger import logger

from lib.core.model.head.ssd_head import SSDHead

from lib.core.anchor.anchor import anchor_tools

class DSFD():

    def __init__(self,):
        if cfg.MODEL.fpn:
            self.ssd_backbone=shufflenet_v2_fpn                 ### it is a func
        else:
            self.ssd_backbone=shufflenet_v2                 ### it is a func
        self.ssd_head=SSDHead()                         ### it is a class

    def forward(self,inputs,boxes,labels,l2_regulation,training_flag,with_loss=True):

        ###preprocess
        inputs=self.preprocess(inputs)

        ### extract feature maps
        origin_fms=self.ssd_backbone(inputs,training_flag)

        reg, cls = self.ssd_head(origin_fms, l2_regulation, training_flag,ratios_per_pixel=2)

        ### calculate loss
        reg_loss, cls_loss = ssd_loss(reg, cls, boxes, labels, 'focal_loss')

        ###### adjust the anchors to the image shape, but it trains with a fixed h,w


        ###adaptive anchor
        # h = tf.shape(inputs)[1]
        # w = tf.shape(inputs)[2]
        # anchors_ = get_all_anchors_fpn(max_size=[h, w])
        #


        ###fix anchor
        anchors_ = anchor_tools.anchors /np.array([cfg.DATA.win,cfg.DATA.hin,cfg.DATA.win,cfg.DATA.hin])
        anchors_decode_ = anchor_tools.decode_anchors /np.array([cfg.DATA.win,cfg.DATA.hin,cfg.DATA.win,cfg.DATA.hin])/5.

        self.postprocess(reg, cls, anchors_, anchors_decode_)

        return reg_loss,cls_loss

    def preprocess(self,image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            image= image-127.
        return image

    def postprocess(self, box_encodings, cls, anchors, anchors_decode):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """

        with tf.name_scope('postprocessing'):
            boxes = batch_decode_fix(box_encodings, anchors, anchors_decode)
            # if the images were padded we need to rescale predicted boxes:

            # it has shape [batch_size, num_anchors, 4]
            # scores = tf.nn.softmax(cls, axis=2)  ##ignore the bg

            scores = tf.split(cls, axis=2, num_or_size_splits=2)[1]


            # it has shape [batch_size, num_anchors,class]
            # labels = tf.argmax(scores, axis=2)
            # it has shape [batch_size, num_anchors]

            # scores = tf.reduce_max(scores, axis=2)
            # it has shape [batch_size, num_anchors]
            # scores = tf.expand_dims(scores, axis=-1)
            # it has shape [batch_size, num_anchors]
        boxes = tf.identity(boxes, name='boxes')
        scores = tf.identity(scores, name='scores')

        res = tf.concat([boxes, scores], axis=2)
        res = tf.identity(res, name='outputs')
        return res
