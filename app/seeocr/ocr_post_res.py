#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file ocr_post_res.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-10-24 11:24

import pickle
import os
import json
import cv2
import numpy as np

from seeocr.utils.oss import coss3_put, coss3_domain # noqa
from seeocr.utils.logger import EasyLogger as logger
from seeocr.utils.errcodes import HandlerError
from seeocr.utils import rmdir_p


def_ocr_code = '''
results = {}
for i in range(len(ocr_res) - 1):
    reskey = ocr_res[i][0].strip()
    resval = ocr_res[i + 1][0].strip()
    for key in ocr_keys:
        if key in reskey:
            for val in ocr_keys:
                if val in resval:
                    resval = resval.replace(val, '')
                    break
            results[key] = resval
            break
'''


def ocr_result(pigeon, progress_cb=None):
    if 'cache_path' not in pigeon:
        raise HandlerError(82001, 'not found cache_path')

    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']
    pigeon['task'] = 'seeocr.post'

    if not os.path.isdir(pigeon['cache_path']):
        raise HandlerError(82002, f'cache_path[{cache_path}] cannot open!')

    def _send_progress(x):
        if progress_cb:
            pigeon['progress'] = round(60 + 0.4 * x, 2)
            progress_cb(pigeon)
            logger.info(f"{round(x, 2)} {pigeon['progress']}")

    _send_progress(2)
    with open(f'{cache_path}/rec_res.pkl', 'rb') as fr:
        ocr_res = pickle.load(fr)
    with open(f'{cache_path}/config.json', 'r') as fr:
        config = json.load(fr)
    ocr_keys = config.get('ocr_keys')
    ocr_code = config.get('process_code')
    _G_, _L_ = {'ocr_keys': ocr_keys, 'ocr_res': ocr_res}, {}
    if len(ocr_code) == 0:
        ocr_code = def_ocr_code
    _send_progress(30)
    try:
        exec(ocr_code, _G_, _L_)
        json_result = _L_['results']
        logger.info(f'{json_result}')
    except Exception:
        raise HandlerError(82005, f'exec ocr_ocde fail!')

    with open(f'{cache_path}/result.json', 'w') as fw:
        json.dump(json_result, fw, indent=4, ensure_ascii=False)
    pigeon['upload_files'].append('result.json')
    pigeon['ocr_res_json'] = f'{coss3_domain}{coss3_path}/result.json'

    if devmode:
        det_points = np.asarray(np.load(f'{cache_path}/det_points.npy'))
        source_path = pigeon['source_path']
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        for pts in det_points:
            box = pts.reshape((-1, 1, 2)).astype(np.int64)
            image = cv2.polylines(image, [box], True, (255, 0, 0), 2)
        cv2.imwrite(f'{cache_path}/det_boxes.png', image)
        pigeon['upload_files'].append('det_boxes.png')
        pigeon['draw_det_boxes'] = f'{coss3_domain}{coss3_path}/det_boxes.png'

    _send_progress(90)
    if coss3_path:
        prefix_map = [cache_path, coss3_path]
        for fn in pigeon['upload_files']:
            coss3_put(f'{cache_path}/{fn}', prefix_map)
    pigeon.pop('upload_files')
    _send_progress(100)
    rmdir_p(os.path.dirname(cache_path))
    logger.info(f'{pigeon}')
    return None
