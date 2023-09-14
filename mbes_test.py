from collections import defaultdict
import copy
import json
import torch
import logging
import numpy as np
import argparse
import open3d as o3d
import sys
import os
from easydict import EasyDict as edict
from tqdm import tqdm
from torch.utils.data import DataLoader

from param import *
from models import *

from torch_geometric.data import Data
from mbes_data.datasets.mbes_data import get_multibeam_datasets
from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm, ransac_pose_estimation
from mbes_data.lib.evaluations import update_results, save_results_to_file
setup_seed(0)

def draw_results(data, pred_trans):
    gt_trans = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(data['points_src'])

    src_pcd_trans = to_o3d_pcd(data['points_src'])
    src_pcd_trans.transform(pred_trans)
    print(f'pred transform: {pred_trans}')

    src_pcd_gt = to_o3d_pcd(data['points_src'])
    src_pcd_gt.transform(gt_trans)
    print(f'gt trans: {gt_trans}')

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(data['points_ref'])

    src_pcd_trans.paint_uniform_color([1, 0, 0])
    src_pcd_gt.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries(
        [src_pcd, ref_pcd, src_pcd_trans, src_pcd_gt])

def custom_collate_fn(list_data):
  if len(list_data) == 0:
    return None
  return list_data[0]

def test(config):
    # Load data
    config.dataset_type = 'multibeam_npy_for_bathynn'
    _, _, test_set = get_multibeam_datasets(config)
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        shuffle=False)

    # Load model
    model = MatcherTest().to(config.device)
    model_dict = torch.load(MODEL_PATH)
    model.load_model(model_dict)
    assert model.training == False
    model.keypoint_thresh = config.keypoint_thresh

    results = defaultdict(dict)
    outdir = os.path.join(config.exp_dir, f'{MODEL_PATH}/kpthresh_{config.keypoint_thresh}')
    os.makedirs(outdir, exist_ok=True)

    for i, data in tqdm(enumerate(test_loader), total=len(test_set)):
        if data is None:
            print(f'Idx {i} is None! Skipping...')
            continue
        
        # Convert data to bathynn format
        data_src = Data(pos=data['data_src'].reshape(-1, 6))
        data_tgt = Data(pos=data['data_tgt'].reshape(-1, 6))
        src_tgt_pose = Data(pos=data['src_tgt_pose'].reshape(-1, 6))
        indices = Data(x=data['indices'].reshape(-1, 2))

        data_bathynn = [data_src, data_tgt, src_tgt_pose, indices]
        for item in data_bathynn:
            item = item.to(config.device)
        output = model(data_bathynn[0], data_bathynn[1])

        eval_data = data['sample']
        for k, v in eval_data.items():
            if isinstance(v, torch.Tensor):
                eval_data[k] = v.squeeze(dim=0)
        if len(output) == 1:
            logging.info(f'idx = {i}: No keypoints found in one of the data')
            eval_data['success'] = False
            results = update_results(results, eval_data, np.eye(4),
                                     config, outdir, logging.getLogger())
        else:
            assert len(output) == 6
            data_points, data_keypoints, feat_anc, feat_pos, abs1, abs2 = output
            print(f'kps_anc: {data_keypoints["anc"].shape}, kps_pos: {data_keypoints["pos"].shape}')
            print(f'feat_anc: {feat_anc.shape}, feat_pos: {feat_pos.shape}')
            eval_data['feat_src_points'] = abs1
            eval_data['feat_ref_points'] = abs2
            eval_data['feat_src'] = feat_anc
            eval_data['feat_ref'] = feat_pos
            ransac_result = ransac_pose_estimation(abs1,
                                                abs2,
                                                feat_anc,
                                                feat_pos,
                                                mutual=False,
                                                distance_threshold=config.voxel_size*1.5,
                                                ransac_iterations=config.ransac_iterations,)
            pred_trans = ransac_result.transformation
            eval_data['success'] = True
            results = update_results(results, eval_data, pred_trans,
                                     config, outdir, logging.getLogger())

            if config.draw_registration_results:
                draw_results(eval_data, pred_trans)
    # save results of the last MBES file
    save_results_to_file(logging.getLogger(), results, config, outdir)


if __name__ == '__main__':
    # Set up logging
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        handlers=[ch])

    logging.basicConfig(level=logging.INFO, format="")

    # Load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--mbes_config',
                        type=str,
                        default='mbes_data/configs/mbesdata_test_meters.yaml',
                        help='Path to multibeam data config file')
    parser.add_argument('--network_config',
                        type=str,
                        default='network_configs/kpthresh02.yaml',
                        help='Path to network config file')
    args = parser.parse_args()
    mbes_config = edict(load_config(args.mbes_config))
    network_config = edict(load_config(args.network_config))
    print(f'MBES data config: {mbes_config}')
    print(f'Network config: {network_config}')

    config = copy.deepcopy(mbes_config)
    for k, v in network_config.items():
        if k not in config:
            config[k] = v
    if config.use_gpu:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    test(config)
