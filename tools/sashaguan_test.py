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
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default='../configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_sashaguan.py', help='test config file path')
    parser.add_argument('--checkpoint', default='../data/sashaguan/sashaguan_result/latest.pth', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--zdf_dir', type=str, default='../data/sashaguan/zdf/',
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
        '--show-dir',default='../data/sashaguan/sashaguan_test/', help='directory where painted images will be saved')
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

def get_all_xyz(xy_t, zdf_path):
    """
    通过给定的二维点坐标list，找出其对应的三维坐标。
    :param xy_t: type：list，一堆二维点坐标。
    :param zdf_path: type：str，zdf文件的路径
    :return: type：list， xyz二维点对应的三维坐标。nanxyz二维点对应到三维坐标中如果是nan值则记录对应二维坐标的索引值。
    """
    app = zivid.Application() #不能删除
    #print(f"Reading point cloud from file: {zdf_path}")
    frame = zivid.Frame(zdf_path)
    #print("Getting point cloud from frame")
    point_cloud = frame.point_cloud()
    xyz = point_cloud.copy_data("xyz")
    xyz = np.array(xyz, dtype=float)
    x_test = xyz[xy_t[:, 1], xy_t[:, 0], 0]
    x_test = np.expand_dims(x_test, axis=1)
    y_test = xyz[xy_t[:, 1], xy_t[:, 0], 1]
    y_test = np.expand_dims(y_test, axis=1)
    z_test = xyz[xy_t[:, 1], xy_t[:, 0], 2]
    z_test = np.expand_dims(z_test, axis=1)
    xyz_test = np.hstack((x_test, y_test, z_test))

    return xyz_test

def get_point_in_img(img):
    """
    从分割网络的预测图像中，取出不同标签的像素值坐标。
    :param img:预测网络输出的预测图像
    :return:
    """
    img_shape = np.array(img)
    h, w = img_shape.shape[0], img_shape.shape[1]
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


    return guidao_point, sashaguan_point, tamian_point


if __name__ == '__main__':
    arg = parse_args()
    zdfs_path = arg.zdf_dir
    zdfs_path_list = []
    img_pre_list = seg()#2d分割网络的预测出的图像
    with open('../data/sashaguan/test.txt', 'r') as ff:
        for line in ff.readlines():
            zdfs_path_list.append(line.split('\n')[0])

    for i in range(len(img_pre_list)):
        img = img_pre_list[i]
        img1 = img.astype('uint8')
        zdf_dir = zdfs_path + str(zdfs_path_list[i]) + '.zdf'
        guidao_2dpoint, sashaguan_2dpoint, tamian_2dpoint = get_point_in_img(img1)#分割后的轨道，撒沙管，踏面的像素点，并进行裁剪，减少像素值。

        #使用2d像素点的坐标，映射到3d，获取3d的点云信息
        guidao_3dpoint = get_all_xyz(guidao_2dpoint, zdf_dir)
        sashaguan_3dpoint = get_all_xyz(sashaguan_2dpoint, zdf_dir)
        tamian_3dpoint = get_all_xyz(tamian_2dpoint, zdf_dir)

        #使用open3d将array转为pcd数据
        guidao_pcd = o3d.geometry.PointCloud()
        guidao_pcd.points = o3d.utility.Vector3dVector(guidao_3dpoint)
        sashaguan_pcd = o3d.geometry.PointCloud()
        sashaguan_pcd.points = o3d.utility.Vector3dVector(sashaguan_3dpoint)
        tamian_pcd = o3d.geometry.PointCloud()
        tamian_pcd.points = o3d.utility.Vector3dVector(tamian_3dpoint)

        # 对点云数据进行均匀下采样。
        guidao_r_pcd = o3d.geometry.PointCloud.uniform_down_sample(guidao_pcd, 30)
        sashaguan_r_pcd = o3d.geometry.PointCloud.uniform_down_sample(sashaguan_pcd, 30)
        tamian_r_pcd = o3d.geometry.PointCloud.uniform_down_sample(tamian_pcd, 30)

        # 对pcd数据进行统计滤波，去除孤点。
        num_points = 10
        radius = 2
        guidao_ror_pcd, _ = guidao_r_pcd.remove_statistical_outlier(num_points, radius)
        sashaguan_ror_pcd, _ = sashaguan_r_pcd.remove_statistical_outlier(num_points, radius)
        tamian_ror_pcd, _ = tamian_r_pcd.remove_statistical_outlier(num_points, radius)

        # 对点云数据使用半径滤波
        # num_points = 20
        # radius = 2.5
        # guidao_ror_pcd, _ = guidao_pcd.remove_radius_outlier(num_points, radius)
        # sashaguan_ror_pcd, _ = sashaguan_pcd.remove_radius_outlier(num_points, radius)
        # tamian_ror_pcd, _ = tamian_pcd.remove_radius_outlier(num_points, radius)

        # 对点云数据使用体素滤波
        # voxel_size = 5
        # guidao_ror_pcd = guidao_pcd.voxel_down_sample(voxel_size)
        # sashaguan_ror_pcd = sashaguan_pcd.voxel_down_sample(voxel_size)
        # tamian_ror_pcd = tamian_pcd.voxel_down_sample(voxel_size)

        o3d.io.write_point_cloud("guidao{}.pcd".format(i), guidao_ror_pcd)
        o3d.io.write_point_cloud("sashaguan_ror_pcd{}.pcd".format(i), sashaguan_ror_pcd)
        o3d.io.write_point_cloud("tamian_ror_pcd{}.pcd".format(i), tamian_ror_pcd)

        # 分别计算撒沙管与轨面与踏面的距离
        sashaguan_guidao_dis = sashaguan_ror_pcd.compute_point_cloud_distance(guidao_ror_pcd)
        sashaguan_tamian_dis = sashaguan_ror_pcd.compute_point_cloud_distance(tamian_ror_pcd)

        f = open(arg.show_dir + '{}.txt'.format(zdfs_path_list[i].split('.')[0]), 'w')
        f.write('撒沙管距轨面距离: ' + str(np.min(sashaguan_guidao_dis)) + "mm" + "\n" +'撒沙管距踏面距离: ' +
                str(np.min(sashaguan_tamian_dis)) + "mm")
        f.close()