
import random
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as run_checkpoint
import numpy as np
import cv2

import data
import networks
import utils


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed : {seed}")


def data_loader(data_cfg, device):
    if data_cfg.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = data.load_llff_data(data_cfg.datadir, data_cfg.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=data_cfg.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, data_cfg.datadir)
        #                   (20,756,1008,3) (120, 3, 5) [ 756.     1008.      815.1316]
        if not isinstance(i_test, list):
            i_test = [i_test]

        if data_cfg.llffhold > 0:
            print('Auto LLFF holdout,', data_cfg.llffhold)
            i_test = np.arange(images.shape[0])[::data_cfg.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if data_cfg.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif data_cfg.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = data.load_blender_data(data_cfg.datadir, 
                                                                        data_cfg.half_res, 
                                                                        data_cfg.testskip, 
                                                                        data_cfg.num_render_poses)
        print('Loaded blender', images.shape, render_poses.shape, hwf, data_cfg.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if data_cfg.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif data_cfg.dataset_type == 'dtu':
        images, poses, render_poses, hwf, i_split = data.load_dtu_data(scene=data_cfg.shape,
                                                                basedir=data_cfg.datadir,
                                                                testskip=data_cfg.testskip)

        print('Loaded dtu', images.shape, render_poses.shape, hwf, data_cfg.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        raise Exception('Unknown dataset type', data_cfg.dataset_type, 'exiting')

    return i_train, i_val, i_test, torch.Tensor(images).to(device), torch.Tensor(poses).to(device), torch.Tensor(render_poses).to(device), hwf, near, far 


def baseline_model_load(model_cfg, device):
    model = {}
    parameter = []
    
    embed_fn, input_ch = utils.get_embedder(model_cfg.multires, model_cfg.i_embed)
    
    input_ch_views = 0
    embeddirs_fn = None
    if model_cfg.use_viewdirs:
        embeddirs_fn, input_ch_views = utils.get_embedder(model_cfg.multires_views, model_cfg.i_embed)
    output_ch = 5 if model_cfg.N_importance > 0 else 4
    skips = [4]
    
    #coarse network
    model['coarse'] = networks.NeRF(D=model_cfg.netdepth, W=model_cfg.netwidth,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=model_cfg.use_viewdirs)

    #fine network
    if model_cfg.N_importance > 0:
        model['fine'] = networks.NeRF(D=model_cfg.netdepth_fine, W=model_cfg.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=model_cfg.use_viewdirs)

    if model_cfg.load_weight:
        print("Loading Network weights")
        for key in model.keys():
            file = os.path.join(model_cfg.weight_path, f'{key}.pth')
            if os.path.isfile(file):
                print(f"Success load {key} weight")
                model_load_dict = torch.load(file, map_location=device)
                model[key].load_state_dict(model_load_dict)
            else:
                print(f"Dose not exist {file}")
            
    for key, val in model.items():
        model[key] = nn.DataParallel(val)
        model[key].to(device)
        model[key].train()
        parameter += list(val.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : utils.run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=model_cfg.netchunk_per_gpu*torch.cuda.device_count())
    
    if model_cfg.checkpoint_rendering:
        # Pass a dummy input tensor that requires grad so checkpointing does something
        # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/10
        dummy = torch.ones(1, dtype=torch.float32, requires_grad=True, device=device)
        network_fn_wrapper = lambda x, y:  model['coarse'](x)
        network_fine_wrapper = lambda x, y: model['fine'](x)
        network_fn = lambda x: run_checkpoint(network_fn_wrapper, x, dummy)
        network_fine = lambda x: run_checkpoint(network_fine_wrapper, x, dummy)
    else:
        network_fn = model['coarse']
        network_fine = model['fine']

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : model_cfg.perturb,
        'N_importance' : model_cfg.N_importance,
        'network_fine' : network_fine,
        'N_samples' : model_cfg.N_samples,
        'network_fn' : network_fn,
        'use_viewdirs' : model_cfg.use_viewdirs,
        'white_bkgd' : model_cfg.white_bkgd,
        'raw_noise_std' : model_cfg.raw_noise_std,
        'alpha_act_fn' : F.softplus if model_cfg.use_softplus_alpha else F.relu
    }
    
    # test 
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    
    return  model, parameter, render_kwargs_train, render_kwargs_test

def get_rel_depths(path, total_num_view):
    rel_depths_list = []

    for i in range(total_num_view):
        rel_depth = cv2.imread(f'{path}/depth{i}.png', -1) / 1000.0
        # shift = rel_depth.min()
        # scale = rel_depth.max() - shift
        # rel_depths_list.append(torch.from_numpy((rel_depth - shift)/scale).type(torch.FloatTensor))
        rel_depths_list.append(torch.from_numpy(rel_depth).type(torch.FloatTensor))
    
    rel_depths = torch.stack(rel_depths_list, 0)
    return rel_depths