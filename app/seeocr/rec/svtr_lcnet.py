#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file rec.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-10-21 16:06


import numpy as np
import pickle
import os

from seeocr.utils.logger import EasyLogger as logger
from seeocr.utils.errcodes import HandlerError

REC_CKPTS_PATH = '/ckpts/rec_svtr_lcnet/Student'
predictor, input_tensor, output_tensor = None, None, None


def _get_predict_instance():
    # paddle(cuda) must promise initiate in the same progress
    from paddle import inference
    rec_config = inference.Config(f'{REC_CKPTS_PATH}/inference.pdmodel', f'{REC_CKPTS_PATH}/inference.pdiparams')
    rec_config.enable_use_gpu(500, 0)
    # rec_config.disable_gpu()
    rec_config.enable_memory_optim()
    rec_config.disable_glog_info()
    rec_config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    rec_config.delete_pass("matmul_transpose_reshape_fuse_pass")
    rec_config.switch_use_feed_fetch_ops(False)
    rec_config.switch_ir_optim(True)
    predictor = inference.create_predictor(rec_config)
    input_tensor = predictor.get_input_handle(predictor.get_input_names()[0])
    output_tensor = predictor.get_output_handle(predictor.get_output_names()[0])
    return predictor, input_tensor, output_tensor


def ocr_recognize(pigeon, progress_cb=None):
    global predictor, input_tensor, output_tensor
    if predictor is None:
        predictor, input_tensor, output_tensor = _get_predict_instance()

    from .augimg import seeocr_rec_transforms
    from .postprocess import seeocr_rec_postprocess

    if 'cache_path' not in pigeon:
        raise HandlerError(82001, 'not found cache_path')

    cache_path = pigeon['cache_path']
    pigeon['task'] = 'seeocr.rec'

    if not os.path.isdir(pigeon['cache_path']):
        raise HandlerError(82002, f'cache_path[{cache_path}] cannot open!')

    def _send_progress(x):
        if progress_cb:
            pigeon['progress'] = round(40 + 0.2 * x, 2)
            progress_cb(pigeon)
            logger.info(f"{round(x, 2)} {pigeon['progress']}")

    with open(f'{cache_path}/det_boxes.pkl', 'rb') as fr:
        det_boxes = pickle.load(fr)
    boxes_num = len(det_boxes)
    width_list = []
    for box in det_boxes:
        width_list.append(box.shape[1] / float(box.shape[0]))

    indices = np.argsort(np.array(width_list))
    rec_res = [['', 0.0]] * len(det_boxes)
    batch_num = pigeon['rec_batch_num']
    rec_image_shape = pigeon['rec_image_shape']
    rec_wh_ratio = rec_image_shape[2] / rec_image_shape[1]

    for beg_img_no in range(0, boxes_num, batch_num):
        end_img_no = min(boxes_num, beg_img_no + batch_num)
        max_wh_ratio = rec_wh_ratio
        for ino in range(beg_img_no, end_img_no):
            h, w = det_boxes[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)

        norm_img_batch = []
        for ino in range(beg_img_no, end_img_no):
            norm_img = seeocr_rec_transforms(det_boxes[indices[ino]], rec_image_shape, max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        input_tensor.copy_from_cpu(norm_img_batch)
        predictor.run()
        output = output_tensor.copy_to_cpu()
        rec_result = seeocr_rec_postprocess(output)
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]
    logger.info(f'{rec_res}')
    return None
