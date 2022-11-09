#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file db.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-10-20 19:58

from paddle import inference
import time
import numpy as np
import pickle
import cv2
import os
import json


from seeocr.utils.easydict import DotDict
from seeocr.utils.logger import EasyLogger as logger
from seeocr.utils.errcodes import HandlerError # noqa
from seeocr.utils import mkdir_p, rmdir_p, easy_wget # noqa
from seeocr.utils.draw import get_rect_points

from .augimg import seeocr_det_transforms
from .postprocess import seeocr_det_postprocess


DET_CKPTS_PATH = '/ckpts/det_db/Student'

det_config = inference.Config(f'{DET_CKPTS_PATH}/inference.pdmodel', f'{DET_CKPTS_PATH}/inference.pdiparams')
det_config.disable_gpu()
det_config.enable_memory_optim()
det_config.disable_glog_info()
det_config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
det_config.delete_pass("matmul_transpose_reshape_fuse_pass")
det_config.switch_use_feed_fetch_ops(False)
det_config.switch_ir_optim(True)

predictor = inference.create_predictor(det_config)
input_tensor = predictor.get_input_handle(predictor.get_input_names()[0])
output_tensor = predictor.get_output_handle(predictor.get_output_names()[0])


def ocr_detect(args, progress_cb=None):
    if 'dev_args' in args and len(args['dev_args']) > 0:
        args.update(json.loads(args['dev_args']))

    args = DotDict(args)
    ocr_keys = args.get('ocr_keys', None)
    if ocr_keys is None or len(ocr_keys) == 0:
        raise HandlerError(80011, f'ocr_keys is None!')

    devmode = args.get('devmode', False)
    resdata = {'errno': 0, 'pigeon': args.pigeon, 'devmode': devmode, 'task': 'seeocr.det', 'upload_files': []}

    def _send_progress(x):
        if progress_cb:
            resdata['progress'] = round(0.4 * x, 2)
            progress_cb(resdata)
            logger.info(f"{round(x, 2)} {resdata['progress']}")

    _send_progress(1)

    suffix = args.source[-3:]
    if 'https://' in args.source:
        segs = args.source[8:].split('/')
        vname = segs[-1].split('.')[0]
        coss3_path = os.path.join('/', *segs[1:-2], 'outputs', vname, 'seeocr')
    else:
        vname = 'unknow'
        coss3_path = '' # noqa

    source_path = args.source
    logger.info(f'from: {source_path}')

    cache_path = f'/data/cache/{int(time.time() * 1000)}/{vname}'
    mkdir_p(cache_path)
    resdata['cache_path'] = cache_path

    if not os.path.isfile(source_path):
        try:
            if suffix == 'mp4':
                source_path = easy_wget(source_path, f'{cache_path}/source.mp4')
            else:
                source_path = easy_wget(source_path, f'{cache_path}/source.png')
        except Exception as err:
            raise HandlerError(80001, f'wget [{args.source}] fail [{err}]!')
    resdata['source_path'] = source_path

    if suffix == 'mp4':
        raise NotImplementedError

    max_side_len = args.get('det_max_side_len', 320)
    thresh = args.get('det_thresh', 0.3)
    box_thresh = args.get('det_box_thresh', 0.6)
    unclip_ratio = args.get('det_unclip_ratio', 1.5)

    _send_progress(20)
    img_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
    height, width = img_bgr.shape[:2]

    black_box = get_rect_points(width, height, args.get('black_box', [0, 0, 0, 0]))
    if black_box is not None:
        black_x1, black_y1, black_x2, black_y2 = black_box
        img_bgr[black_y1:black_y2, black_x1:black_x2, :] = 0
    focus_box = get_rect_points(width, height, args.get('focus_box', [0, 0, 1, 1]))
    if focus_box is not None:
        focus_x1, focus_y1, focus_x2, focus_y2 = focus_box
        img_bgr = img_bgr[focus_y1:focus_y2, focus_x1:focus_x2, :]
        cv2.imwrite(source_path, img_bgr)

    image, shape = seeocr_det_transforms(img_bgr, max_side_len)

    logger.info(f'{image.shape}, {shape.tolist()}')

    images = np.expand_dims(image, axis=0)
    shapes = np.expand_dims(shape, axis=0)

    _send_progress(30)
    input_tensor.copy_from_cpu(images.copy())
    predictor.run()
    det_outs = output_tensor.copy_to_cpu()

    _send_progress(80)
    boxes, points = seeocr_det_postprocess(det_outs, shapes, img_bgr, thresh, box_thresh, unclip_ratio)

    resdata['det_boxes_count'] = len(boxes)
    if len(boxes) == 0:
        logger.error('no boxes detect')
        resdata['progress'] = 100
        if progress_cb:
            progress_cb(resdata)
        rmdir_p(os.path.dirname(cache_path))
        return None

    with open(f'{cache_path}/det_boxes.pkl', 'wb') as fw:
        pickle.dump(boxes, fw)

    with open(f'{cache_path}/config.json', 'w') as f:
        f.write(json.dumps(dict(args)))

    if devmode:
        np.save(f'{cache_path}/det_points.npy', points)
    resdata['upload_files'].append('config.json')
    resdata['coss3_path'] = coss3_path
    resdata['rec_batch_num'] = args.get('rec_batch_num', 6)
    resdata['rec_image_shape'] = args.get('rec_image_shape', (3, 48, 320))

    _send_progress(100)
    return resdata
