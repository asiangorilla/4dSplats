from scene.scene_read import searchForMaxIteration,add_points,readColmapSceneInfo
from gaussian_model.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams

import os



class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 路径，迭代次数
        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化训练相机，测试相机，视频相机列表
        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        # 场景类型判定
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print('Reading Colmap Scene Info...')
            # --> SceneInfo Class: pcd，cams，ply path，maxtime
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
        #     dataset_type="blender"
        # elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
        #     scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
        #     dataset_type="dynerf"
        # elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
        #     scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
        #     dataset_type="nerfies"
        # elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
        #     scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
        #     dataset_type="PanopticSports"
        # elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
        #     scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
        #     dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type! Only colmap projects are supported "

        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        print("Dataset type: ", dataset_type)
        self.cameras_extent = scene_info.nerf_normalization["radius"] # cameras_extent --> 相机能够捕捉到的最大场景半径

        # 获取训练相机，测试相机，视频相机
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)


        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))

        # self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            # self.gaussians.load_model(os.path.join(self.model_path,
            #                                         "point_cloud",
            #                                         "iteration_" + str(self.loaded_iter),
            #                                        ))
        else:
            # 根据前面的scene_info中的point_cloud,初始化GaussianModel
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    # save point cloud and deformation
    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # self.gaussians.save_deformation(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera