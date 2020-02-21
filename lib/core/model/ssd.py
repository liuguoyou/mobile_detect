#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

from lib.core.anchor.box_utils import batch_decode,batch_decode_fix

from lib.core.model.net.shufflenet.shufflenet import shufflenet_v2_ssd
from lib.core.model.net.mobilenetv3.backbone import mobilenetv3_ssd
from lib.core.model.net.mobilenet.backbone import mobilenet_ssd
from lib.core.model.loss.ssd_loss import ssd_loss
from train_config import config as cfg

from lib.helper.logger import logger

from lib.core.model.head.ssd_head import SSDHead

from lib.core.anchor.anchor import anchor_tools

class mobile_ssd():

    def __init__(self,):
        if "ShufflenetV2"  in cfg.MODEL.net_structure:
            self.ssd_backbone=shufflenet_v2_ssd                 ### it is a func
        elif "MobilenetV2" in cfg.MODEL.net_structure:
            self.ssd_backbone = mobilenet_ssd
        elif "MobilenetV3" in cfg.MODEL.net_structure:
            self.ssd_backbone = mobilenetv3_ssd
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

        ###adaptive anchor, more time consume
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


            mean = cfg.DATA.PIXEL_MEAN
            std = np.asarray(cfg.DATA.PIXEL_STD)
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_invstd = tf.constant(1./std, dtype=tf.float32)
            image = (image - image_mean) * image_invstd  ###imagenet preprocess just centered the data

            # image= image-127.
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
            scores = tf.nn.sigmoid(scores)

        if "tf" ==cfg.MODEL.deployee:
            ###this branch is for tf
            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')

        elif "coreml" ==cfg.MODEL.deployee:
            ###this branch is for coreml

            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')

        elif "mnn" == cfg.MODEL.deployee:

            ##this branch is for mnn
            boxes = tf.squeeze(boxes, axis=[0])
            scores = tf.squeeze(scores, axis=[0, 2])

            selected_indices = tf.image.non_max_suppression(
                boxes, scores, cfg.MODEL.max_box, cfg.MODEL.iou_thres, cfg.MODEL.score_thres)
            #
            boxes = tf.gather(boxes, selected_indices)
            scores = tf.gather(scores, selected_indices)

            num_boxes = tf.cast(tf.shape(boxes)[0], dtype=tf.int32)
            zero_padding = cfg.MODEL.max_box - num_boxes

            boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])

            scores = tf.expand_dims(scores, axis=-1)
            scores = tf.pad(scores, [[0, zero_padding], [0, 0]])


            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')

        else:
            ###this branch is for tf

            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')


        return tf.concat([boxes,scores],axis=-1)

