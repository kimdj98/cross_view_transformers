import pathlib

import torch
import torchvision
import numpy as np

from PIL import Image
from .common import encode, decode
from .augmentations import StrongAug, GeometricAug


class Sample(dict):
    def __init__(
        self,
        token,
        scene,
        intrinsics,
        extrinsics,
        images,
        view,
        bev,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene

        self.view = view
        self.bev = bev

        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics,
            'prev_images': batch.prev_images,
            'prev_intrinsics': batch.prev_intrinsics,
            'prev_extrinsics': batch.prev_extrinsics
        }

    def get_bev(self, batch: Sample):
        result = {
            'view': batch.view,
        }

        scene_dir = self.labels_dir / batch.scene

        bev_path = f'bev_{batch.token}.png'
        Image.fromarray(encode(batch.bev)).save(scene_dir / bev_path)

        result['bev'] = bev_path

        # Auxilliary labels
        if batch.get('aux') is not None:
            aux_path = f'aux_{batch.token}.npz'
            np.savez_compressed(scene_dir / aux_path, aux=batch.aux)

            result['aux'] = aux_path

        # Visibility mask
        if batch.get('visibility') is not None:
            visibility_path = f'visibility_{batch.token}.png'
            Image.fromarray(batch.visibility).save(scene_dir / visibility_path)

            result['visibility'] = visibility_path

        # velocity_map
        if batch.get('velocity_map') is not None:
            velocity_map_path = f'velocity_map_{batch.token}.npz'
            np.savez_compressed(scene_dir / velocity_map_path, velocity_map=batch.velocity_map)

            result['velocity_map'] = velocity_map_path

        return result

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_bev(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result


class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, dataset_dir, labels_dir, image_config, num_classes, augment='none'):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.labels_dir = pathlib.Path(labels_dir)
        self.image_config = image_config
        self.num_classes = num_classes

        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        self.to_tensor = super().__call__

    def get_cameras(self, sample: Sample, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()

        for image_path, I_original in zip(sample.images, sample.intrinsics):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop

            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))

        prev_images = list()
        prev_intrinsics = list()

        for prev_image_path, prev_I_original in zip(sample.prev_images, sample.prev_intrinsics):
            h_resize = h + top_crop
            w_resize = w

            prev_image = Image.open(self.dataset_dir / prev_image_path)

            prev_image_new = prev_image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            prev_image_new = prev_image_new.crop((0, top_crop, prev_image_new.width, prev_image_new.height))

            prev_I = np.float32(prev_I_original)
            prev_I[0, 0] *= w_resize / prev_image.width
            prev_I[0, 2] *= w_resize / prev_image.width
            prev_I[1, 1] *= h_resize / prev_image.height
            prev_I[1, 2] *= h_resize / prev_image.height
            prev_I[1, 2] -= top_crop

            prev_images.append(self.img_transform(prev_image_new))
            prev_intrinsics.append(torch.tensor(prev_I))

        return {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),

            'prev_image': torch.stack(prev_images, 0),
            'prev_intrinsics': torch.stack(prev_intrinsics, 0),
            'prev_extrinsics': torch.tensor(np.float32(sample.prev_extrinsics)),
        }

    def get_bev(self, sample: Sample):
        scene_dir = self.labels_dir / sample.scene
        bev = None

        if sample.bev is not None:
            bev = Image.open(scene_dir / sample.bev)
            bev = decode(bev, self.num_classes)
            bev = (255 * bev).astype(np.uint8)
            bev = self.to_tensor(bev)

        result = {
            'bev': bev,
            'view': torch.tensor(sample.view),
        }

        if 'visibility' in sample:
            visibility = Image.open(scene_dir / sample.visibility)
            result['visibility'] = np.array(visibility, dtype=np.uint8)

        if 'aux' in sample:
            aux = np.load(scene_dir / sample.aux)['aux']
            result['center'] = self.to_tensor(aux[..., 1])

        if 'pose' in sample:
            result['pose'] = np.float32(sample['pose'])

        return result
    
    def get_state(self, sample: Sample):
        
        if 'past_coordinate' in sample:
            past_coordinate = torch.tensor(np.float32(sample['past_coordinate']))

        if 'past_vel' in sample:
            past_vel = torch.tensor(np.float32(sample['past_vel']))

        if 'past_acc' in sample:
            past_acc = torch.tensor(np.float32(sample['past_acc']))

        if 'past_yaw' in sample:
            past_yaw = torch.tensor(np.float32(sample['past_yaw']))

        if 'label_waypoint' in sample:
            label_waypoint = torch.tensor(np.float32(sample['label_waypoint']))

        if 'label_vel' in sample:
            label_vel = torch.tensor(np.float32(sample['label_vel']))

        if 'label_acc' in sample:
            label_acc = torch.tensor(np.float32(sample['label_acc']))
        
        if 'label_yaw' in sample:
            label_yaw = torch.tensor(np.float32(sample['label_yaw']))

        result = {
            'past_coordinate': past_coordinate,
            'past_vel': past_vel,
            'past_acc': past_acc,
            'past_yaw': past_yaw,
            'label_waypoint': label_waypoint,
            'label_vel': label_vel,
            'label_acc': label_acc,
            'label_yaw': label_yaw,
        }

        return result
    
    # get velocity map
    def get_velocity_map(self, sample: Sample):
        scene_dir = self.labels_dir / sample.scene
        velocity_map = None

        if sample.velocity_map is not None:
            velocity_map = np.load(scene_dir / sample.velocity_map)['velocity_map']
            velocity_map = self.to_tensor(velocity_map)

        result = {
            'velocity_map': velocity_map,
        }

        return result

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)

        result = dict()
        result.update(self.get_cameras(batch, **self.image_config))
        result.update(self.get_bev(batch))
        result.update(self.get_state(batch))
        result.update(self.get_velocity_map(batch))

        return result
