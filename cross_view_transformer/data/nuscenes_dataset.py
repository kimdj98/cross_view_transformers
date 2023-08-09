import torch
import numpy as np
import cv2
import math

from pathlib import Path
from functools import lru_cache

from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon

from .common import INTERPOLATION, get_view_matrix, get_pose, get_split
from .transforms import Sample, SaveDataTransform

from nuscenes.prediction import PredictHelper
from nuscenes.nuscenes import NuScenes

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

VEHICLE = ['car', 'truck', 'bus', 'trailer', 'construction', 'motorcycle', 'bicycle']

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    dataset='unused',                   # ignore
    augment='unused',                   # ignore
    image='unused',                     # ignore
    label_indices='unused',             # ignore
    num_classes=NUM_CLASSES,            # in here to make config consistent
    **dataset_kwargs
):
    assert num_classes == NUM_CLASSES

    helper = NuScenesSingleton(dataset_dir, version)
    transform = SaveDataTransform(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    result = list()

    for scene_name, scene_record in helper.get_scenes():
        if scene_name not in split_scenes:
            continue

        data = NuScenesDataset(scene_name, scene_record, helper,
                               transform=transform, **dataset_kwargs)
        result.append(data)

    return result


class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.map_expansion.map_api import NuScenesMap

        map_name = self.nusc.get('log', log_token)['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton


class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def __init__(
        self,
        scene_name: str,
        scene_record: dict,
        helper: NuScenesSingleton,
        transform=None,
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
    ):
        self.scene_name = scene_name
        self.transform = transform

        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])

        self.view = get_view_matrix(**bev)
        self.bev_shape = (bev['h'], bev['w'])

        self.samples = self.parse_scene(scene_record, cameras)

    def parse_scene(self, scene_record, camera_rigs):
        data = []
        sample_token = scene_record['first_sample_token']

        for i in range(scene_record['nbr_samples'] - 14): # for each sample in the scene (except the last 12)
        # while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)
    
    def parse_sample_record(self, sample_record, camera_rig):
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)

        # Calculate ego pose in lidar frame
        orig = np.array(egolidar['translation'][:2])

        # Calculate original heading angle (yaw)
        q = Quaternion(egolidar['rotation'])
        yaw, _, _ = q.yaw_pitch_roll
        orig_yaw = math.degrees(yaw)

        angle_in_radian = (np.pi / 2) - yaw
        rot_matrix = np.array([[np.cos(angle_in_radian), -np.sin(angle_in_radian)],
                               [np.sin(angle_in_radian), np.cos(angle_in_radian)]], dtype=np.float32)

        past_coordinate = np.zeros((5,2))
        past_vel        = np.zeros((5,2))
        past_acc        = np.zeros((5,2))
        past_yaw        = np.zeros((5,1))

        label_waypoint  = np.zeros((14,2)) # later cut off to 12 time steps
        has_pred        = np.zeros((14,1)) 
        label_vel       = np.zeros((13,2)) # later cut off to 12 time steps
        label_acc       = np.zeros((12,2)) 
        label_yaw       = np.zeros((14,1))  

        # get ego states
        # careful of thinking prev as past token, prev token is future time step token
        # timestep+0 <- timestep+1 <- timestep+2
        # curr_token <- prev_token <- pprev_token
        #      i     <-    i-1     <-     i-2
        pprev_token = None
        prev_token = None
        curr_token = sample_record['token']
        for i in range(5): # curr: i, prev: i-1, pprev: i-2
            if not curr_token:
                break
            
            # get current sample data
            curr_sample = self.nusc.get('sample', curr_token)
            lidar_data  = self.nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            curr_ego_token = lidar_data['ego_pose_token']
            curr_coordinate = self.nusc.get('ego_pose', curr_ego_token)['translation'][:2] - orig
            curr_coordinate = rot_matrix @ curr_coordinate
            curr_yaw = math.degrees(Quaternion(self.nusc.get('ego_pose', curr_ego_token)['rotation']).yaw_pitch_roll[0]) - orig_yaw

            # get ego coordinate (relative to current ego coordinate)
            past_coordinate[i, :] = curr_coordinate
            # get ego yaw (relative to current ego yaw)
            past_yaw[i, :] = curr_yaw

            if prev_token:
                # get +1 time step sample data
                prev_sample = self.nusc.get('sample', prev_token)
                prev_lidar_data = self.nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])
                prev_ego_token = prev_lidar_data['ego_pose_token']

                # get ego velocity
                prev_vel = (past_coordinate[i-1, :] - curr_coordinate) / 0.5
                past_vel[i-1, :] = prev_vel

            
            if pprev_token:
                # get +2 time step sample data
                pprev_sample = self.nusc.get('sample', pprev_token)
                pprev_lidar_data = self.nusc.get('sample_data', pprev_sample['data']['LIDAR_TOP'])
                pprev_ego_token = pprev_lidar_data['ego_pose_token']

                # get ego acceleration
                pprev_acc = (past_vel[i-2, :] - prev_vel) / 0.5
                past_acc[i-2, :] = pprev_acc
            
            # update step tokens
            pprev_token = prev_token
            prev_token = curr_token
            curr_token = self.nusc.get('sample', curr_token)['prev']

        curr_token = sample_record['token']
        curr_token = self.nusc.get('sample', curr_token)['next']

        # get labels
        for i in range(14):
            if not curr_token:
                break
            
            # get current sample data
            curr_sample = self.nusc.get('sample', curr_token)
            lidar_data  = self.nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            curr_ego_token = lidar_data['ego_pose_token']
            curr_coordinate = self.nusc.get('ego_pose', curr_ego_token)['translation'][:2] - orig
            curr_coordinate = rot_matrix @ curr_coordinate
            curr_yaw = math.degrees(Quaternion(self.nusc.get('ego_pose', curr_ego_token)['rotation']).yaw_pitch_roll[0]) - orig_yaw
            
            label_waypoint[i, :] = curr_coordinate
            label_yaw[i, :] = curr_yaw
            has_pred[i, :] = 1
            curr_token = self.nusc.get('sample', curr_token)['next']

            if (i-1) >= 0:
                label_vel[i-1, :] = (label_waypoint[i, :] - label_waypoint[i-1, :]) / 0.5
            
            if (i-2) >= 0:
                label_acc[i-2, :] = (label_vel[i-1, :] - label_vel[i-2, :]) / 0.5

        assert has_pred.all() == 1, "all waypoints should have prediction"

        return {
            # sample info
            'scene': self.scene_name,
            'token': sample_record['token'],

            # ego pose info
            'pose': world_from_egolidarflat.tolist(),
            'pose_inverse': egolidarflat_from_world.tolist(),

            # camera info
            'cam_ids': list(camera_rig),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
            
            # state info
            'past_coordinate': past_coordinate.tolist(),
            'past_vel': past_vel.tolist(),
            'past_acc': past_acc.tolist(),
            'past_yaw': past_yaw.tolist(),

            # label info
            'label_waypoint': label_waypoint[:12].tolist(),
            'label_vel':      label_vel[:12].tolist(),
            'label_acc':      label_acc[:12].tolist(),
            'label_yaw':      label_yaw[:12].tolist(),
        }

    def get_dynamic_objects(self, sample, annotations):
        h, w = self.bev_shape[:2]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            segmentation[mask] = 255
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (sigma ** 2))

            # orientation, h/2, w/2
            center_ohw[mask, 0:2] = ((front - center) / (np.linalg.norm(front - center) + 1e-6))[None]
            center_ohw[mask, 2:3] = np.linalg.norm(front - center)
            center_ohw[mask, 3:4] = np.linalg.norm(left - center)

            visibility[mask] = ann['visibility_token']

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]

        result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

        # (h, w, 1 + 1 + 2 + 2)
        return result, visibility

    def convert_to_box(self, sample, annotations):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.bottom_corners()                                              # 3 4
            center = corners.mean(-1)                                                   # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7

            yield p                                                                     # 3 7

    def convert_to_velocity_box(self, sample, annotations):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.bottom_corners()                                              # 3 4
            center = corners.mean(-1)                                                   # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7
            
            # calculate velocity using center of vehicle
            if a["prev"]:
                prev = self.nusc.get('sample_annotation', a["prev"])
                prev_box = data_classes.Box(prev['translation'], prev['size'], Quaternion(prev['rotation']))

                prev_corners = prev_box.bottom_corners()                                # 3 4
                prev_center = prev_corners.mean(-1)                                     # 3

                velocity = center - prev_center                                         # 3

            else:
                velocity = np.zeros(3)                                                  # 3


            yield p, velocity                                                                 # 3 7


    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None

    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result

    def get_line_layers(self, sample, layers, patch_radius=150, thickness=1):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)
                line = self.nusc_map.extract_line(polygon_token['line_token'])

                p = np.float32(line.xy)                                     # 2 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
                p = V @ S @ M_inv @ p                                       # 3 n
                p = p[:2].round().astype(np.int32).T                        # n 2

                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_static_layers(self, sample, layers, patch_radius=150):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map.extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
                    exteriors = [V @ S @ M_inv @ p for p in exteriors]
                    exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
                    interiors = [V @ S @ M_inv @ p for p in interiors]
                    interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_dynamic_layers(self, sample, anns_by_category):
        h, w = self.bev_shape[:2]
        result = list()

        for anns in anns_by_category:
            render = np.zeros((h, w), dtype=np.uint8)

            # fill in the polygons
            for p in self.convert_to_box(sample, anns):
                p = p[:2, :4] # 2 4

                cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_velocity_layers(self,
                            sample, 
                            vehicle_annotations, 
                            debug=False # debug keyword for visualization
                            ): 
        h, w = self.bev_shape[:2]
        result = list()

        if debug:
            visualization_map = np.ones((h, w, 3), dtype=np.uint8)*255 # set background to white

        v_x_map = np.zeros((h, w))
        v_y_map = np.zeros((h, w))
        render = np.zeros((h, w))

        # fill in the polygons
        for p, velocity in self.convert_to_velocity_box(sample, vehicle_annotations):
            p = p[:2, :4] # 2 4

            velocity = velocity[:2] * 2 # multiply by 2 since the fps is 2Hz

            # convert velocity into polar coordinates
            magnitude, angle = cv2.cartToPolar(velocity[0], velocity[1])

            mask = cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)
            v_x_map += mask * velocity[0] # for every annotation in the sample, add x velocity to the map
            v_y_map += mask * velocity[1] # for every annotation in the sample, add y velocity to the map

            if debug: # if debug mode is on, visualize the velocity in 2d BEV
                magnitude = magnitude[:1] * 100 # multiply by 100 to make the visualization clearer
                angle = angle[:1]

                # normalize angle values to range 0-360
                hue = ((angle / (2 * np.pi)) * 360).astype(np.uint8)

                # set saturation/value to max
                saturation = np.full_like(hue, 255)
                value = np.full_like(hue, magnitude)

                hsv = cv2.merge([hue, saturation, value])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                bgr = bgr[0][0]

                color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

                cv2.fillPoly(visualization_map, [p.round().astype(np.int32).T], color, INTERPOLATION) # visualize the velocity in 2d BEV
                # DEBUG: print veloicty magnitude color
                # print(f"======================================")
                # print(f"velocity: ({velocity[0]}, {velocity[1]})")
                # print(f"magnitude: {magnitude}, angle: {hue}")
                # print(f"color: {color}")
                

        if debug:
            cv2.imwrite(f"{sample['scene']}_{sample['token']}.png", visualization_map)

        v_map = np.stack([v_x_map, v_y_map], -1)

        return v_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Raw annotations
        anns_dynamic = self.get_annotations_by_category(sample, DYNAMIC) 
        anns_vehicle = self.get_annotations_by_category(sample, ['vehicle'])[0]

        static = self.get_static_layers(sample, STATIC)                             # 200 200 2 lane, road_segment
        dividers = self.get_line_layers(sample, DIVIDER)                            # 200 200 2 road_divider, lane_divider
        dynamic = self.get_dynamic_layers(sample, anns_dynamic)                     # 200 200 8 car, truck, bus, trailer, construction, pedestrian, motorcycle, bicycle
        velocity_map = self.get_velocity_layers(sample, anns_vehicle, debug=False)   # 200 200 2 vehicles

        bev = np.concatenate((static, dividers, dynamic), -1)                       # 200 200 12
        assert bev.shape[2] == NUM_CLASSES

        # Additional labels for vehicles only.
        aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=bev,
            velocity_map=velocity_map,
            aux=aux,
            visibility=visibility,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
