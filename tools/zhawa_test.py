# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import zivid
import numpy as np
import math
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default='../configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_zhawa.py', help='test config file path')
    parser.add_argument('--checkpoint', default='../data/zhawa/zhawa_result/latest.pth', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--zdf_dir', type=str, default='../data/zhawa/zdf/HD_zhawa/',
        help=('The path of the ZDF file when measuring'))
    parser.add_argument(
        '--beta', type=int, default=50,
        help=('if specified, the evaluat'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir',default='../data/zhawa/zhawa_test/', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def seg():
    """
    用于分割闸瓦
    :return:type:list, [preimg1[], preimg2[], ...]. 返回每张测试图片中每个像素的预测值。
    """
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)

    return results

def get_4_point(img):
    """
    通过cv2中的函数，获得目标图像的最小外接矩形的四个点坐标及目标区域中所有像素点的坐标
    :param img_path: 语义分割后，只包含一个预测区域的图像
    :return: box1:[[x1, y1], [x2, y2]...]shape为（4，2）的数组，包含四个坐标值; aa:[[x1, y1], [x2, y2], ...]目标区域中所有像素点坐标
    """
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:  # 遍历轮廓
        rect = cv2.minAreaRect(c)  # 生成最小外接矩形
        box_ = cv2.boxPoints(rect)
        box1 = np.int0(box_)
        a = []
        for j in range(min(box1[:, 0]), max(box1[:, 0]) + 1):
            for i in range(min(box1[:, 1]), max(box1[:, 1]) + 1):
                if img[i, j] > 0:
                    a.append([j, i])
        aa = np.array(a)
    return aa, box1

def dist(point1, point2):
    """
    用于2维坐标的距离计算，point1和point2的距离
    :param point11: type: list, len = 1, [x1, y1]
    :param point22: type: list, len = 1, [x2, y2]
    :return: distt type: float, 两点距离
    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x1x2 = x1 - x2
    y1y2 = y1 - y2
    distt = math.sqrt(x1x2 ** 2 + y1y2 ** 2)
    return distt

def point_updown(point_4):
    """
    通过外接矩形的四个点算出两条长边的坐标，并返回两条长边的各三分点的坐标
    :param point_4:  [[x1, y1], [x2, y2]...]shape为（4，2）的数组，包含四个坐标值
    :return: type: arry. point3_up, point3_down，分别包含三个坐标。shape(3,2),是分割轮廓上三个三分点的坐标
    """
    point1 = [point_4[0, 0], point_4[0, 1]]
    point2 = [point_4[1, 0], point_4[1, 1]]
    point3 = [point_4[2, 0], point_4[2, 1]]
    point4 = [point_4[3, 0], point_4[3, 1]]
    d_x1x2 = dist(point1, point2)
    d_x2x3 = dist(point2, point3)
    d_x3x4 = dist(point3, point4)
    d_x4x1 = dist(point4, point1)
    d_all = d_x1x2 + d_x2x3 + d_x3x4 +d_x4x1
    P1_x, P1_y = point1[0], point1[1]
    P2_x, P2_y = point2[0], point2[1]
    P3_x, P3_y = point3[0], point3[1]
    P4_x, P4_y = point4[0], point4[1]

    xy_up = []
    xy_down = []

    #用于判断外接矩形的边是否是长边
    if d_x1x2 > (d_all / 4):
        x1_up = P2_x + ((P1_x - P2_x) / 4)
        x2_up = P2_x + ((P1_x - P2_x) / 2)
        x3_up = P2_x + ((3 * P1_x - 3 * P2_x) / 4)
        y1_up = (((x1_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
        y2_up = (((x2_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
        y3_up = (((x3_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
        if (P1_x - P2_x) > 0:
            xy_up = [[x1_up, y1_up], [x2_up, y2_up], [x3_up, y3_up]]
        if (P1_x - P2_x) < 0:
            xy_up = [[x3_up, y3_up], [x2_up, y2_up], [x1_up, y1_up]]

    # 用于判断外接矩形的边是否是长边
    if d_x2x3 > (d_all / 4):
        x1_up = P3_x + ((P2_x - P3_x) / 4)
        x2_up = P3_x + ((P2_x - P3_x) / 2)
        x3_up = P3_x + ((3 * P2_x - 3 * P3_x) / 4)
        y1_up = (((x1_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
        y2_up = (((x2_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
        y3_up = (((x3_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
        if (P2_x - P3_x) > 0:
            xy_up = [[x1_up, y1_up], [x2_up, y2_up], [x3_up, y3_up]]
        if (P2_x - P3_x) < 0:
            xy_up = [[x3_up, y3_up], [x2_up, y2_up], [x1_up, y1_up]]

    # 用于判断外接矩形的边是否是长边
    if d_x3x4 > (d_all / 4):
        x1_down = P4_x + ((P3_x - P4_x) / 4)
        x2_down = P4_x + ((P3_x - P4_x) / 2)
        x3_down = P4_x + ((3 * P3_x - 3 * P4_x) / 4)
        y1_down = (((x1_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
        y2_down = (((x2_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
        y3_down = (((x3_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
        if (P3_x - P4_x) > 0:
            xy_down = [[x1_down, y1_down], [x2_down, y2_down], [x3_down, y3_down]]
        if (P3_x - P4_x) < 0:
            xy_down = [[x3_down, y3_down], [x2_down, y2_down], [x1_down, y1_down]]

    # 用于判断外接矩形的边是否是长边
    if d_x4x1 > (d_all / 4):
        x1_down = P1_x + ((P4_x - P1_x) / 4)
        x2_down = P1_x + ((P4_x - P1_x) / 2)
        x3_down = P1_x + ((3 * P4_x - 3 * P1_x) / 4)
        y1_down = (((x1_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
        y2_down = (((x2_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
        y3_down = (((x3_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
        if (P4_x - P1_x) > 0:
            xy_down = [[x1_down, y1_down], [x2_down, y2_down], [x3_down, y3_down]]
        if (P4_x - P1_x) < 0:
            xy_down = [[x3_down, y3_down], [x2_down, y2_down], [x1_down, y1_down]]

    xyint3_up = np.round(np.array(xy_up))
    xyint3_down = np.round(np.array(xy_down))

    return xyint3_up, xyint3_down

def get_point2_incontourrs(point_up, point_down, point_contourrs, bata = 50.0):
    """
    给定目标轮廓外接矩形上对应的一对点，建立一条通过两点的直线。输出目标区域与直线首先接触的点和最后离开点的坐标。
    :param point_up: type: list, [x1, y1]外接矩形，其中一条长边上的某一点，本程序中是长边上的三分三分点
    :param point_down: type: list, [x1, y1]外接矩形，另一条长边上的对应点
    :param point_contourrs: type：arry, [[x1, y1], [x2, y2],...]目标区域中所有点集的集合
    :param bata: 这个值根据图像宽度调整，约为宽度乘2.57%。
    :return: type: arry, [[x1, y1], [x2, y2]]目标区域上对应两点的坐标, shape = (2,)
    """
    x1, y1 = point_up[0], point_up[1]
    x2, y2 = point_down[0], point_down[1]
    numb= point_contourrs.shape[0]
    point_in_contourrs = np.array(point_contourrs)

    #经过给定的两个点，建立一个方程，(y - y1)(x2 - x1) - (x - x1)(y2 - y1) = mbx
    xishu = np.repeat(np.expand_dims(np.array([(y1 - y2), (x2 - x1)]), axis=0), numb, axis=0)
    cc = np.repeat(np.expand_dims(np.array([(x2 * y1 - x2 * y2), (y2 * x2 - y2 * x1)]), axis=0), numb, axis=0)
    ax = np.multiply(point_in_contourrs, xishu)
    bx = ax - cc
    mbx = bx.sum(axis=1)
    mbx = np.expand_dims(mbx, axis=1)
    hh, ll = mbx.shape
    point2_contourrs = []
    for hhh in range(hh):
        if mbx[hhh, 0] < bata:
            if mbx[hhh, 0] > -bata:
                #找出所建立直线附近点的坐标，把坐标放入point2_contourrs中
                pointss = point_in_contourrs[hhh, :]
                point2_contourrs.append(pointss)
    point2_contourrs = np.array(point2_contourrs)

    #把直线附近点的坐标x和y相加，用于找出其和的最大值和最小值
    pointssum = point2_contourrs.sum(axis=1)
    pointssum = np.expand_dims(pointssum, axis=1)
    hh_max, ll_max = np.where(pointssum == np.max(pointssum))
    hh_min, ll_min = np.where(pointssum == np.min(pointssum))
    pointss_max = point2_contourrs[hh_max[0], :]
    pointss_min = point2_contourrs[hh_min[0], :]

    #把点对应到img图像时注意x值对应图像W，y值对应图像H
    return pointss_max, pointss_min

def get_xyz_for_zdf(xyint, zdf_path):
    """
	从zdf文件中获取xyz信息
	xyint：type: list, 2d图像中点的xy坐标，xyint[x, y]
	zdf_path：zdf的路径信息
	return: type: list, 2D图像中点对应的zdf中的xyz坐标xyz_cloud[x, y, z]
	"""
    app = zivid.Application() #不能删除
    #print(f"Reading point cloud from file: {zdf_path}")
    frame = zivid.Frame(zdf_path)
    #print("Getting point cloud from frame")
    point_cloud = frame.point_cloud()
    xyz = point_cloud.copy_data("xyz")
    xyz = np.array(xyz, dtype=float)
    xyz_cloud = []
    for i in range(3):
        xyz_cloud.append(xyz[int(xyint[1]), int(xyint[0]), i])

    return xyz_cloud

def get_all_xyz(xy_t, zdf_path):
    """
    通过给定的二维点坐标list，找出其对应的三维坐标。
    :param xy_t: type：list，一堆二维点坐标。
    :param zdf_path: type：str，zdf文件的路径
    :return: type：list， xyz二维点对应的三维坐标。nanxyz二维点对应到三维坐标中如果是nan值则记录对应二维坐标的索引值。
    """
    xyz = []
    nanxyz = []
    ii = 0
    for i in range(len(xy_t)):
        xy = xy_t[i]
        a = get_xyz_for_zdf(xy, zdf_path)
        ii += 1
        if True in np.isnan(np.array(a)):
            nanxyz.append(i)
        else:
            xyz.append([a[0], a[1], a[2]])
    return xyz, nanxyz

def circle_pixel(point, radius=2):
    """
    给定一个点，在这个点的xy方向分别增加radius个像素值
    :param point: type：list，给定一个点的二维坐标
    :param radius: type：int，某个方向增加啊的像素点个数
    :return: type：list，增加后的像素坐标列表。
    """
    x = int(point[0])
    y = int(point[1])
    list = []
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius +1):
            list.append((i, j))
    return list

def xyz_dist(point11, point22):
    """
    计算两个三维坐标列表的的距离
    :param point11: type：list，[[x1,y1, z1], [x2, y2, z2],...]
    :param point22: type：list
    :return:
    """
    ddd = 0
    for i in range(len(point11)):
        point1 = point11[i]
        disttt = 0
        for j in range(len(point22)):
            point2 = point22[j]
            x1 = point1[0]
            y1 = point1[1]
            z1 = point1[2]
            x2 = point2[0]
            y2 = point2[1]
            z2 = point2[2]
            x1x2 = x1 - x2
            y1y2 = y1 - y2
            z1z2 = z1 - z2
            distt = math.sqrt(x1x2**2 + y1y2**2 + z1z2**2)
            disttt = disttt + distt
        ddd = ddd + disttt
    return ddd/(len(point11)*len(point22))

def measure(point_up_1, point_down_1, zdf_path, contourrs_point):
    dis1 = 0
    zero_times = 0
    for i in range(3):
        point_up, point_down = get_point2_incontourrs(point_up_1[i, :], point_down_1[i, :], contourrs_point)#通过外接矩形上的点获取语义分割后图像上对应点。
        point_up_x1 = point_up.tolist()
        point_down_x1 = point_down.tolist()

        point_up_x1all = circle_pixel(point_up_x1, radius=1)#获取2d中一个点周围的九个点坐标
        point_down_x1all = circle_pixel(point_down_x1, radius=1)

        a, _ = get_all_xyz(point_up_x1all, zdf_path)#使用这九个点去影射3D中对应的点云数码
        b, _ = get_all_xyz(point_down_x1all, zdf_path)
        if len(a)*len(b) != 0:#除去2d对应到3d时，3d中点云数据为nan时的情况。
            dis = xyz_dist(a, b)
            dis1 = dis1 + dis
        else:
            zero_times += 1
            print("zero_times", zero_times)
    if zero_times == 3:
        zhawa_size = 'Measurement failed'
    else:
        zhawa_size = str(dis1/(3 - zero_times))#将三次测量的距离求均值。
    return zhawa_size

def get_point_in_img(img):
    """
    从分割网络的预测图像中，取出不同标签的像素值坐标。
    :param img:预测网络输出的预测图像
    :return:
    """
    img_shape = np.array(img)
    h, w = img_shape.shape[0], img_shape.shape[1]
    print('....................', h)
    guidao_point = []
    sashaguan_point = []
    tamian_point = []
    guidao_point_h = []
    sashaguan_point_h = []
    tamian_point_h = []

    for i in range(h):
        for j in range(w):
            if img[i][j] == 1:
                guidao_point.append([j, i])
            elif img[i][j] == 2:
                sashaguan_point.append([j, i])
            elif img[i][j] == 3:
                tamian_point.append([j, i])

    guidao_point = np.array(guidao_point)
    sashaguan_point = np.array(sashaguan_point)
    tamian_point = np.array(tamian_point)

    sashaguan_point_max = np.expand_dims(np.amax(sashaguan_point, axis=0), axis=0)
    sashaguan_point_min = np.expand_dims(np.amin(sashaguan_point, axis=0), axis=0)
    sashaguan_point_mid = np.round((sashaguan_point_max - sashaguan_point_min) * 0.5)
    sashaguan_point_mid = sashaguan_point_mid + sashaguan_point_min
    print('sashaguan_point_mid', sashaguan_point_mid)

    for i in range(sashaguan_point.shape[0]):
        if sashaguan_point[i, 0] >= sashaguan_point_mid[0, 0]:
            sashaguan_point_h.append(sashaguan_point[i, :])

    for i in range(guidao_point.shape[0]):
        if guidao_point[i, 0] >= sashaguan_point_mid[0, 0] and guidao_point[i, 0] <= sashaguan_point_max[0, 0]:
            guidao_point_h.append(guidao_point[i, :])

    for i in range(tamian_point.shape[0]):
        if tamian_point[i, 1] >= sashaguan_point_min[0, 1] and tamian_point[i, 1] <= sashaguan_point_max[0, 1]:
            tamian_point_h.append(tamian_point[i, :])

    sashaguan_point_h = np.array(sashaguan_point_h)
    guidao_point_h = np.array(guidao_point_h)
    tamian_point_h = np.array(tamian_point_h)

    sashaguan_point_h10 = []
    guidao_point_h10 = []
    tamian_point_h10 = []
    ii = 0
    jj = 0
    zz = 0
    for i in range(round((sashaguan_point_h.shape[0])/10)):
        sashaguan_point_h10.append(sashaguan_point_h[ii, :])
        ii += 10
    for i in range(round((guidao_point_h.shape[0])/10)):
        guidao_point_h10.append(guidao_point_h[jj, :])
        jj += 10
    for i in range(round((tamian_point_h.shape[0])/10)):
        tamian_point_h10.append(tamian_point_h[zz, :])
        zz += 10
    sashaguan_point_h10 = np.array(sashaguan_point_h10)
    guidao_point_h10 = np.array(guidao_point_h10)
    tamian_point_h10 = np.array(tamian_point_h10)
    print('sashaguan_point_h10', sashaguan_point_h10.shape)
    print('sashaguan_point_h10', guidao_point_h10.shape)
    print('sashaguan_point_h10', tamian_point_h10.shape)
    print('sashaguan_point_h', sashaguan_point_h.shape)

    # imgg = np.zeros((1200, 1944), dtype=np.uint8)
    # for i in range(sashaguan_point_h10.shape[0]):
    #     imgg[sashaguan_point_h10[i, 1], sashaguan_point_h10[i, 0]] = 170
    #
    # for i in range(guidao_point_h10.shape[0]):
    #     imgg[guidao_point_h10[i, 1], guidao_point_h10[i, 0]] = 85
    #
    # for i in range(tamian_point_h10.shape[0]):
    #     imgg[tamian_point_h10[i, 1], tamian_point_h10[i, 0]] = 255
    #
    # cv2.namedWindow('aa', 0)
    # cv2.resizeWindow('aa', 600, 400)
    # cv2.imshow('aa', imgg)
    # cv2.waitKey(0)

    print('s_g_t_shape', sashaguan_point_h.shape, guidao_point_h.shape, tamian_point_h.shape)

    return guidao_point_h10, sashaguan_point_h10, tamian_point_h10


if __name__ == '__main__':
    arg = parse_args()
    zdfs_path = arg.zdf_dir
    zdfs_path_list = []
    img_pre_list = seg()

    with open('../data/zhawa/test.txt', 'r') as ff:
        for line in ff.readlines():
            zdfs_path_list.append(line.split('\n')[0])

    for i in range(len(img_pre_list)):
        img = img_pre_list[i]
        img1 = img.astype('uint8')
        point_in_contourrs, point_4 = get_4_point(img1)
        point_up_1, point_down_1 = point_updown(point_4)
        zdf_dir = zdfs_path + str(zdfs_path_list[i]) + '.zdf'
        print(".............")
        zhawa = measure(point_up_1, point_down_1, zdf_dir, point_in_contourrs)
        f = open(arg.show_dir + '{}.txt'.format(zdfs_path_list[i].split('.')[0]), 'w')
        f.write('zhawa hou du: ' + str(zhawa))
        f.close()
        print('img', img.shape)