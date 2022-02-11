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
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default='../configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_chegou.py', help='test config file path')
    parser.add_argument('--checkpoint', default='../data/chegou/result/latest.pth', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--zdf_dir', type=str, default='../data/sashagaun/zdf/HD_zhawa/',
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
        '--show-dir',default='../data/chegou/test/', help='directory where painted images will be saved')
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
    用于分割车钩和轨面
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

def measure_touying(point_chegou2d, point_guimian2d, zdf_path):
    """
    使用投影法测量车钩与轨面的距离，在空间坐标系中，将车钩中心线与轨面上y轴值相同的点取出来，
    计算两点x轴上的差值，这个差值为车钩中心线到轨面的高度。
    :param point_chegou: type:list 车钩的xy坐标
    :param point_guimian: type:list  轨面的xy坐标
    :param zdf_path: 对应的zdf文件路径
    :return: 车钩距轨面的距离
    """
    point_chegou, _ = get_all_xyz(point_chegou2d, zdf_path)
    point_guimian, _ = get_all_xyz(point_guimian2d, zdf_path)

    point_chegou = np.array(point_chegou)
    point_guimian = np.array(point_guimian)

    print('point_chegou.shape: ', point_chegou.shape)
    guimian_shape1 = point_guimian.shape[0]
    dist_all = []
    dist_times = 0

    for i in range(point_chegou.shape[0]):
        point_chegoui = np.repeat(np.expand_dims(point_chegou[i, :], axis=0), guimian_shape1, axis=0)
        print('point_chegoui.shape', point_chegoui.shape)
        dist_point = point_chegoui - point_guimian
        dist_point = abs(dist_point)
        dist_pointi = dist_point[dist_point[:, 1].argsort()]
        dist_i = 0
        dist_itimes = 0
        dist_itimes_zeros = 0
        for ii in range(dist_pointi.shape[0]):
            print('dist_pointi[ii, :]', dist_pointi[ii, :])
            if dist_pointi[ii, 1] < 10:
                dist_itimes += 1
                dist_i = dist_i + dist_pointi[ii, 0]
            else:
                dist_itimes_zeros += 1
        if dist_itimes != 0:
            dist_alli = dist_i/dist_itimes
        else:
            continue
        dist_all.append(dist_alli)

    return dist_all

def get_point_in_img(img):
    """
    从分割网络的预测图像中，取出不同标签的像素值坐标。
    :param img:预测网络输出的预测图像
    :return:
    """
    img_shape = np.array(img)
    h, w = img_shape.shape[0], img_shape.shape[1]
    print('....................', h)
    chegou_point = []
    guimian_point = []

    chegou_point10 = []
    guimian_point10 = []

    for i in range(h):
        for j in range(w):
            if img[i][j] == 1:
                chegou_point.append([j, i])
            elif img[i][j] == 2:
                guimian_point.append([j, i])

    chegou_point = np.array(chegou_point)
    guimian_point = np.array(guimian_point)

    ii = 0
    jj = 0

    for i in range(round((chegou_point.shape[0])/10)):
        chegou_point10.append(chegou_point[ii, :])
        ii += 10
    for i in range(round((guimian_point.shape[0])/10)):
        guimian_point10.append(guimian_point[jj, :])
        jj += 10

    chegou_point10 = np.array(chegou_point10)
    guimian_point10 = np.array(guimian_point10)
    return chegou_point10, guimian_point10

def measure_sanjiao(point_chegou2d, point_guimian2d, zdf_path):
    """
    使用勾股定理计算车钩中心线到轨面的高度，计算得到车钩中心线和轨面的坐标值，然后计算车钩中心线到轨面的最短距离，
    将这个距离作为直角三角形的斜边，然后计算车钩中心线与轨面在z轴上的差值，作为一条直角边，再使用勾股定理计算出高度。
    :param point_chegou2d: type:list 车钩的xy坐标
    :param point_guimian2d: type:list  轨面的xy坐标
    :param zdf_path: 对应的zdf文件路径
    :return: 车钩据轨面的距离
    """
    point_chegou, _ = get_all_xyz(point_chegou2d, zdf_path)
    point_guimian, _ = get_all_xyz(point_guimian2d, zdf_path)

    point_chegou = np.array(point_chegou)
    point_guimian = np.array(point_guimian)

    chegou_pcd = o3d.geometry.PointCloud()
    chegou_pcd.points = o3d.utility.Vector3dVector(point_chegou)
    guimian_pcd = o3d.geometry.PointCloud()
    guimian_pcd.points = o3d.utility.Vector3dVector(point_guimian)

    dists = chegou_pcd.compute_point_cloud_distance(guimian_pcd)
    distss = 0
    for i in range(len(dists)):
        distss = dists[i] + distss
    dist = distss/len(dists)
    hight = (dist ** 2 - 900 ** 2) ** 0.5
    return hight

def compere_angle(a1, a2):
    data_M = np.sqrt(np.sum(a1 * a1, axis=1))
    data_N = np.sqrt(np.sum(a2 * a2, axis=1))
    cos_theta = np.sum(a1 * a2, axis=1) / (data_M * data_N)
    theta = np.degrees(np.arccos(cos_theta))  # 角点b的夹角值
    return theta

def measure_nihepingmian(point_chegou2d, point_guimian2d, zdf_path, sita = 16):
    """
    对轨面拟合一个平面，计算车钩到这个平面的距离作为车钩据轨面的高度，语义分割出轨面在三维空间中的位置，计算轨面上每个点的法向量，
    给定一个与轨面垂直的单位法向量，筛选轨面上点法向量与给定单位法向量夹角小于8度的点，使用这些点拟合出一个平面。
    :param point_chegou2d: type:list 车钩的xy坐标
    :param point_guimian2d: type:list  轨面的xy坐标
    :param zdf_path: 对应的zdf文件路径
    :param sita: 相机偏离水平方向的角度
    :return: 车钩据轨面的距离
    """
    point_chegou, _ = get_all_xyz(point_chegou2d, zdf_path)
    point_guimian, _ = get_all_xyz(point_guimian2d, zdf_path)

    point_chegou = np.array(point_chegou)
    point_guimian = np.array(point_guimian)
    chegou_pcd = o3d.geometry.PointCloud()
    guimian_pcd = o3d.geometry.PointCloud()
    chegou_pcd.points = o3d.utility.Vector3dVector(point_chegou)
    guimian_pcd.points = o3d.utility.Vector3dVector(point_guimian)

    point_chegou = np.array(chegou_pcd.points)
    point_guimian = np.array(guimian_pcd.points)
    plen = []
    downpcd = guimian_pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))# 计算法线，只考虑邻域内的20个点
    sins = np.sin(np.pi / (180 / sita))
    coss = np.cos(np.pi / (180 / sita))
    print(coss, ' ', sins)
    for i in range(len(downpcd.points)):
        a1 = np.asarray([coss, 0, sins]).reshape(1, 3)
        a2 = np.asarray(downpcd.normals[i]).reshape(1, 3)
        angle = compere_angle(a1, a2)
        if angle[0] < 8:
            plen.append(np.asarray(downpcd.points[i]))
    plen = np.asarray(plen)
    plen_pcd = o3d.geometry.PointCloud()
    plen_pcd.points = o3d.utility.Vector3dVector(plen)
    plane_model, inliers = plen_pcd.segment_plane(distance_threshold=2, ransac_n=round(2 * len(plen_pcd.points)/5), num_iterations=500)
    A, B, C, D = plane_model[0], plane_model[1], plane_model[2], plane_model[3]
    hh = point_chegou.shape[0]
    one_list = [[1] for ii in range(hh)]
    point_chegou = np.append(point_chegou, one_list, axis=1)
    plent_array = np.array([A, B, C, D]).reshape(1, 4)
    print(plent_array)
    plent_array = np.repeat(plent_array, hh, axis=0)
    m_array = np.multiply(plent_array, point_chegou)
    mod_d = np.sum(m_array, axis=1)
    mod_area = np.sqrt(np.sum(np.square([A, B, C])))
    d_all = abs(mod_d) / mod_area
    d = np.sum(d_all)/len(d_all)
    return d

def measure_chegounihepingmian(point_chegou2d, point_guimian2d, zdf_path,):
    point_chegou, _ = get_all_xyz(point_chegou2d, zdf_path)
    point_guimian, _ = get_all_xyz(point_guimian2d, zdf_path)

    point_chegou = np.array(point_chegou)
    point_guimian = np.array(point_guimian)
    chegou_pcd = o3d.geometry.PointCloud()
    guimian_pcd = o3d.geometry.PointCloud()
    chegou_pcd.points = o3d.utility.Vector3dVector(point_chegou)
    guimian_pcd.points = o3d.utility.Vector3dVector(point_guimian)

    chegou_array = point_chegou
    print(chegou_array.shape)
    chegou_arraypai = chegou_array[chegou_array[:, 1].argsort()]
    ind = chegou_arraypai.shape[0]

    p1 = np.asarray(chegou_arraypai[20, :]).reshape(1, 3)
    p2 = np.asarray(chegou_arraypai[round(ind / 2), :]).reshape(1, 3)
    p3 = np.asarray(chegou_arraypai[ind - 25, :]).reshape(1, 3)
    print(p1, p2, p3)
    x1, y1, z1 = p1[0, 0], p1[0, 1], p1[0, 2]
    x2, y2, z2 = p2[0, 0], p2[0, 1], p2[0, 2]
    x3, y3, z3 = p3[0, 0], p3[0, 1], p3[0, 2]
    print(x1, y1, z1)
    print(x2, y2, z2)
    A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    B = (x3 - x1) * (z2 - z1) - (z3 - z1) * (x2 - x1)
    C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    D = -(A * x1 + B * y1 + C * z1)

    model = []
    for y in range(0, 600):
        for z in range(700, 1900):
            x = -(y * B + z * C + D) / A
            model.append([x, y, z])

    model = np.array(model)
    guimian_pcd = o3d.geometry.PointCloud()
    guimian_pcd.points = o3d.utility.Vector3dVector(model)
    o3d.io.write_point_cloud('test1.ply', guimian_pcd)

    hh = point_guimian.shape[0]
    one_list = [[1] for ii in range(hh)]
    point_guimian = np.append(point_guimian, one_list, axis=1)
    plent_array = np.array([A, B, C, D]).reshape(1, 4)
    print(plent_array)
    plent_array = np.repeat(plent_array, hh, axis=0)
    m_array = np.multiply(plent_array, point_guimian)
    mod_d = np.sum(m_array, axis=1)
    mod_area = np.sqrt(np.sum(np.square([A, B, C])))
    d_all = abs(mod_d) / mod_area
    d = np.sum(d_all) / len(d_all)
    return d

if __name__ == '__main__':
    arg = parse_args()
    zdfs_path = arg.zdf_dir
    zdfs_path_list = []
    img_pre_list = seg()
    with open('../data/chegou/test.txt', 'r') as ff:
        for line in ff.readlines():
            zdfs_path_list.append(line.split('\n')[0])

#for i in range(len(img_pre_list)):
    img = img_pre_list[4]
    img1 = img.astype('uint8')
    print('img', img.shape)
    a1, a2 = get_point_in_img(img1)
    print('\n', a1.shape, a2.shape)
    a1, a2 = a1.tolist(), a2.tolist()
    dd = measure_nihepingmian(a1, a2, r'D:\liuxz_code\python\mmsegmentation\data\chegou\zdf\330.zdf', 16)
    ddd = measure_chegounihepingmian(a1, a2, r'D:\liuxz_code\python\mmsegmentation\data\chegou\zdf\330.zdf')
    print(ddd)
    print(dd)