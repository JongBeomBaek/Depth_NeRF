import pdb
import os
import argparse

import yaml
from dotmap import DotMap
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm,  trange
import wandb

import initialize
import utils

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./config/train_llff.yaml")

args = parser.parse_args()

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    with open(args.conf, 'r') as f:
        # configuration
        conf =  yaml.load(f, Loader=yaml.FullLoader)
        train_cfg = DotMap(conf['Train'])
        device = torch.device("cuda" if train_cfg.Base.use_cuda else "cpu")
        consistency_keep_keys = ['disp_map', 'disp0', 'rgb_map', 'rgb0']

        # seed 
        initialize.seed_everything(train_cfg.Base.seed)

        # data loader
        i_train, i_val, i_test, images, poses, render_poses, hwf, near, far = initialize.data_loader(train_cfg.Dataset, device)
        H, W, focal = int(hwf[0]), int(hwf[1]), hwf[-1] #torch.Tensor(hwf[-1:]).to(device)

        # model load
        model, parameters, render_kwargs_train, render_kwargs_test = initialize.baseline_model_load(train_cfg.Model, device)

        # optimizer = torch.optim.Adam([{"params": affine_params, "lr": 0.00001}, 
        #                         {"params": parameters}], args.lrate, betas=(0.9, 0.999))
        optimizer = optim.Adam(parameters, float(train_cfg.Base.lrate), betas=(0.9, 0.999))
        scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.Base.render_autocast)

        if train_cfg.Base.wandb:
            wandb.init(project = "Depth_NeRF",
                        name = train_cfg.Base.expname,
                        config = conf)
            wandb.watch(model['coarse'])

        #relative depth set
        rel_depths = initialize.get_rel_depths(train_cfg.Custom.rel_depth_path, train_cfg.Custom.num_total_views).to(device)
        
        start = 1 
        N_iters = train_cfg.Base.N_iters + 1
        import time
        for i in trange(start, N_iters):
            index = np.random.randint(train_cfg.Base.max_train_views)
            img_i = i_train[index]
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            rays_o, rays_d = utils.get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            rel_depth = rel_depths[img_i]

            # rays_o, rays_d = utils.get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
            batch_rays, select_coords = utils.sample_rays(H, W, rays_o, rays_d, N_rand=train_cfg.Model.N_rand,
                i=i, start=start, precrop_iters=train_cfg.Base.precrop_iters, precrop_frac=train_cfg.Base.precrop_frac)

            batch_rays = batch_rays.to(device)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            target_s = target_s.to(device)
            rel_depth_s = (rel_depth[select_coords[:, 0], select_coords[:, 1]]).to(device)

            # Representational consistency loss with rendered image
            if train_cfg.Base.consistency_loss and i % train_cfg.Base.render_loss_interval == 0:
                with torch.no_grad():
                    # TODO: something strange with pts_W in get_rays when 224 nH
                    # rays = utils.get_rays(H, W, focal, c2w=pose,
                    rays = utils.get_rays(H, W, focal, c2w=torch.Tensor(pose),
                                nH=train_cfg.Base.render_nH, nW=train_cfg.Base.render_nW, jitter=train_cfg.Base.render_jitter_rays)
                with torch.cuda.amp.autocast(enabled=train_cfg.Base.render_autocast):
                    extras = utils.render(H, W, focal, chunk=train_cfg.Base.chunk,
                                    rays=(rays[0].to(device), rays[1].to(device)),
                                    keep_keys=consistency_keep_keys,
                                    **render_kwargs_train)[-1]      
                                  
                    # rgb0 is the rendering from the coarse network, while rgb_map uses the fine network
                    if train_cfg.Model.N_importance > 0:
                        disps = torch.stack([extras['disp_map'], extras['disp0']], dim=0) #torch.Size([2, 168, 168, 3])
                    else:
                        disps = extras['disp_map'].unsqueeze(0)
                    disps = F.interpolate(disps.unsqueeze(1), size=(H, W), mode=train_cfg.Base.pixel_interp_mode)

            #####  Core optimization loop  #####
            optimizer.zero_grad()
            loss = 0
            rgb, disp, acc, extras = utils.render(H, W, focal, train_cfg.Base.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
                
            img_loss = utils.img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss += img_loss
            psnr = utils.mse2psnr(img_loss)

            nan_mask = ~disp.isnan()
            # temp = torch.abs(utils.min_max_norm(rel_depth_s[nan_mask]).detach() - utils.min_max_norm(disp[nan_mask]))
            temp = torch.abs(rel_depth_s[nan_mask].detach() - disp[nan_mask])
            # pdb.set_trace()
            loss += temp.mean()
            # loss += torch.abs(utils.min_max_norm(rel_depth_s[nan_mask]).detach() - utils.min_max_norm(disp[nan_mask])).mean()

            if 'rgb0' in extras:
                if i == start:
                    print('Using auxilliary rgb0 mse loss')
                img_loss0 = utils.img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = utils.mse2psnr(img_loss0)
                # loss += torch.abs(utils.min_max_norm(rel_depth_s[nan_mask]).detach() - utils.min_max_norm(extras['disp0'])).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # NOTE: IMPORTANT!
            new_lrate = utils.update_lr(float(train_cfg.Base.lrate), train_cfg.Base.lrate_decay, optimizer, i)

            
            ##################################################################################################
            
            metrics = {}

            # Print
            if i%train_cfg.Base.print_iters==0:
                # tqdm.write(f"[TRAIN] Iter: {i} iL: {img_loss.item()} dL0: {depth_loss0.item()} dL: {depth_loss.item()}  PSNR: {psnr.item()}")
                tqdm.write(f"[TRAIN] Iter: {i} iL: {img_loss.item()} PSNR: {psnr.item()}")

            # Save_Component
            if i%train_cfg.Base.save_iters == 0:
                utils.save_component(train_cfg.Base.basedir, train_cfg.Base.expname, i, model, optimizer)
            # Save Video
            if i%train_cfg.Base.video_iters==0 and i > 0 : 
                moviebase = utils.save_video(render_poses, hwf, train_cfg.Base.chunk, render_kwargs_test,
                                    train_cfg.Base.basedir, train_cfg.Base.expname, i)
                if train_cfg.Base.wandb:
                    metrics["render_path/rgb_video"] = wandb.Video(moviebase + 'rgb.mp4')
                    metrics["render_path/disp_video"] = wandb.Video(moviebase + 'disp.mp4')
            # Test
            if i%train_cfg.Base.testset_iters==0 and i > 0:
                utils.test_iter(images, poses, i_test,  hwf, train_cfg.Base.chunk, render_kwargs_test,
                                    train_cfg.Base.basedir, train_cfg.Base.expname, i)


            # Log scalars, images and histograms to wandb
            if i%train_cfg.Base.log_iters==0:
                metrics.update({
                    "train/loss": loss.item(),
                    "train/psnr": psnr.item(),
                    "train/mse": img_loss.item(),
                    "train/lrate": new_lrate,
                })
                metrics["gradients/norm_coarse"] = utils.gradient_norm(model['coarse'].parameters())
                if train_cfg.Model.N_importance > 0:
                    metrics["gradients/norm_fine"] = utils.gradient_norm(model['fine'].parameters())
                    metrics["train/psnr0"] = psnr0.item()
                    metrics["train/mse0"] = img_loss0.item()

            if i%train_cfg.Base.log_raw_hist_iters==0 and train_cfg.Base.wandb:
                metrics["train/tran"] = wandb.Histogram(trans.detach().cpu().numpy())

            if i%train_cfg.Base.img_iters==0 and train_cfg.Base.wandb:
                # Log a rendered validation view to Tensorboard
                with torch.no_grad():
                    img_i = i_val[0]
                    target = images[img_i]
                    pose = poses[img_i, :3,:4].to(device)
                    rgb, disp, acc, extras = utils.render(H, W, focal, chunk=train_cfg.Base.chunk,
                                                    c2w=pose, **render_kwargs_test)
                    psnr = utils.mse2psnr(utils.img2mse(rgb, target))

                    metrics = {
                        'val/rgb': wandb.Image(utils.to8b(rgb.cpu().numpy())[np.newaxis]),
                        'val/disp': wandb.Image(disp.cpu().numpy()[np.newaxis,...,np.newaxis]),
                        'val/disp_scaled': utils.make_wandb_image(disp[np.newaxis,...,np.newaxis]),
                        'val/acc': wandb.Image(acc.cpu().numpy()[np.newaxis,...,np.newaxis]),
                        'val/acc_scaled': utils.make_wandb_image(acc[np.newaxis,...,np.newaxis]),
                        'val/psnr_holdout': psnr.item(),
                        'val/rgb_holdout': wandb.Image(target.cpu().numpy()[np.newaxis])
                    }
                    if train_cfg.Model.N_importance > 0:
                        metrics['rgb0'] = wandb.Image(utils.to8b(extras['rgb0'].cpu().numpy())[np.newaxis])
                        metrics['disp0'] = wandb.Image(extras['disp0'].cpu().numpy()[np.newaxis,...,np.newaxis])
                        metrics['z_std'] = wandb.Image(extras['z_std'].cpu().numpy()[np.newaxis,...,np.newaxis])

            if metrics and train_cfg.Base.wandb:
                wandb.log(metrics, step=i)