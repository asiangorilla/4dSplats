from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.colmap_loader import read_points3D_binary, read_points3D_text

from plyfile import PlyData
from typing import NamedTuple
from PIL import Image

import numpy as np
import torch
import math
import sys
import os


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    mask: np.array

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

# READ COLMAP PROJECT
def readColmapSceneInfo(path, images, eval, llffhold=8):
    """
    取并处理COLMAP生成的3D重建场景数据, 包括相机参数、图像信息和点云数据。函数详细处理了数据的读取、排序和分类，并进行了一些基础的数据转换
    :return：
        scene_info：一个封装了场景信息的 SceneInfo 对象，包含以下主要属性：
        point_cloud：点云数据，从PLY文件中读取。
        train_cameras 和 test_cameras：训练和测试用的相机数据。
        video_cameras：通常与训练相机相同，用于视频渲染或其他视觉任务。
        maxtime：场景的时间范围，这里默认设置为0。
        nerf_normalization：场景归一化的参数。
        ply_path：指向点云PLY文件的路径。
    """

    # get extrinsic and intrinsic from path
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # com_infos which has all cameras' config data
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # breakpoint()
    # split train and test in com_infos
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # normalization vector for each cam
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Ensure .ply files are generated and accessible
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        storePly(ply_path, xyz, rgb)  # --> store xyz, rgb into .ply file
    try:
        pcd = fetchPly(ply_path)   # --> read .ply file as BasicPointCloud
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    从COLMAP项目中提取相机的 外参 和 内参 ，以及读取该相机的所有配置参数，包括其对应的单一的图像
    :return
        cam_infos --> 一个包含所有相机信息的列表，该对象封装了一个相机的所有相关数据（包括UID、旋转矩阵、平移向量、视场角、图像路径、图像名、图像宽度和高度等）。
    """
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):

        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.

        cam_infos.append(cam_info)

    sys.stdout.write('\n')

    return cam_infos

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    if np.array(resized_image_PIL).max()!=1:
        resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def storePly(path, xyz, rgb):
    """
    点云的位置和颜色信息格式化并保存到一个PLY文件中

    """

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    """
    read ply from path, extract point cloud info, it contains color, position, ...
    Wrap this into BasicPointCloud object and return
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)




def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getNerfppNorm(cam_info):
    """
    根据一组相机的信息计算归一化参数
    主要帮助设置一个适当的场景坐标系
    使得相机捕捉的场景可以在一个统一且优化的空间尺度中进行处理
    :return
            translate --> translation vector
            radius --> normalized radius
    """

    def get_center_and_diag(cam_centers):
        """
        接收一组相机中心点坐标，用于计算 场景的中心 和 对角线 长度
        """
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()

    return {"translate": translate, "radius": radius}   #


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)

    return np.float32(Rt)




def add_points(pointsclouds, xyz_min, xyz_max):

    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds