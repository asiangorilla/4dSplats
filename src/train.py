import os
import json
import torch
import copy
import numpy as np

from random import randint
from PIL import Image
from tqdm import tqdm
from LossFunction.DeformationLoss import DeformationLoss
from TransformationModel.transModel import DeformationNetworkSeparate, DeformationNetworkConnected
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from LossFunction.utils import o3d_knn, setup_camera, update_params_and_optimizer, params2rendervar, calc_psnr, densify, save_params

def initialize_params(seq, md):
    """
        parameters:
            seq --> Name or Identifier of the dataset
            md --> meta data
        return
            params --> parameters of rendering
            variables --> Auxiliary variables used to calculate rendering
    """
    
    # load data
    init_pt_cld = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]

    max_cams = 50

    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),  # scale factor, computed from knn
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    # transform into pytorch parameters 
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    

    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    
    return params, variables


def initialize_optimizer(params, variables):
    """
        learning rates of parameters
    """
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_dataset(t, md, seq):
    """
        Load and process the dataset for a specific timestep.

        Parameters:
            t: The current timestep index.
            md (dict): The metadata dictionary containing information like image filenames, camera parameters, etc.
            seq (str): The sequence identifier used to locate the data within the file system.

        Returns:
            list: A list of dictionaries, each containing the camera setup, image data, segmentation data, and an identifier for each camera viewpoint.
    """

    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]

        # image into NumPy array
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

        # segmentation image
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))

        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})

    return dataset


def initialize_per_timestep(params, variables, optimizer):
    """
    Parameters:
        params: a dictionary containing all variable parameters in the scene.
        variables: a dictionary containing variables used to keep track of the previous state and other control information.
        optimizer: optimiser object for parameter optimization.

    Returns:
        params: updated dictionary of scene parameters, which are used for scene render
        variables: updated dictionary of auxiliary variables.
    """

    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    # compute new points and rotation
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]

    # update variables
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    # update params and optimizer
    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def get_batch(todo_dataset, dataset):
    """
        Random selection of data samples
    """
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    """
    Initializes and updates variables after the first timestep for dynamic scene processing.

    Parameters:
        params (dict): Dictionary containing current scene parameters such as 3D points, colors, etc.
        variables (dict): Dictionary for storing intermediate states and computed values.
        optimizer (Optimizer): The optimizer object managing learning rates and parameter updates.
        num_knn (int): Number of nearest neighbors to consider in k-NN calculations.

    Returns:
        dict: Updated variables dictionary with new computed values for neighbors and rotational data.
    """

    # divide points of fg and bg
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])

    # compute neighbour information of fg points
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)

    # update variables
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()

    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()

    # fix some parameters
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables



def train(seq, exp):
    """
        parameters:
            seq --> Name or Identifier of the dataset
            exp --> Name of expriment, used to build output path
        output_params:
            --> render parameters, used to do 3d render

    """

    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f" Experiment '{exp}' of dataset --> '{seq}' already exists. Exiting.")
        return
    
    # load metadata 
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata  
    num_timesteps = len(md['fn'])

    # initialize optimizer
    params, variables = initialize_params(seq, md)
    optimizer = initialize_optimizer(params, variables)

    output_params = []

    loss_function = DeformationLoss()

    # training loop
    for t in range(num_timesteps):
        # for each timestep
        dataset = get_dataset(t, md, seq)
        todo_dataset = []

        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)

        num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")

        for i in range(num_iter_per_timestep):
            # for each iteration of one single timestep
            curr_data = get_batch(todo_dataset, dataset)

            # Backward Propagation
            loss, variables = loss_function(params, curr_data, variables, is_initial_timestep)
            loss.backward()

            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)

                if is_initial_timestep:
                    # Adjustment and densification of 3D point cloud data 
                    params, variables = densify(params, variables, optimizer, i)

                # updata params
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))

        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    save_params(output_params, seq, exp)    
