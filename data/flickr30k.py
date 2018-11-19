"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
# from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
from pathlib import Path
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

FLICK_CLASSES = (  # always index 0
    'people', 'clothing', 'bodyparts', 'animals',
    'vehicles', 'instruments', 'scene', 'other')

# FLICK_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
# VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
FLICK_ROOT = '/scratch/arka/Ark_git_files/flickr30k'


class FLICKAnnotationTransform(object):
    """Transforms a FLICK annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of FLICK's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(FLICK_CLASSES, range(len(FLICK_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for k, v in target.items():
            bboxes = v['bbox']
            ent = v['entity']
            if len(bboxes) == 0:
                continue
            for b in bboxes:
                bndbox = []
                for i, pt in enumerate(b):
                    cur_pt = pt - 1
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[ent]
                bndbox.append(label_idx)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        # for obj in target.iter('object'):
        #     difficult = int(obj.find('difficult').text) == 1
        #     if not self.keep_difficult and difficult:
        #         continue
        #     name = obj.find('name').text.lower().strip()
        #     bbox = obj.find('bndbox')

        #     pts = ['xmin', 'ymin', 'xmax', 'ymax']
        #     bndbox = []
        #     for i, pt in enumerate(pts):
        #         cur_pt = int(bbox.find(pt).text) - 1
        #         # scale height or width
        #         cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
        #         bndbox.append(cur_pt)
        #     label_idx = self.class_to_ind[name]
        #     bndbox.append(label_idx)
        #     res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        #     # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class FLICK_Detection(data.Dataset):
    """FLICK Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to FLICKdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'FLICK2007')
    """

    def __init__(self, root, dset='train',
                 transform=None, target_transform=FLICKAnnotationTransform()):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.name = 'flickr30k'
        self.ficd = json.load(open(str(self.root / 'all_ann.json')))
        self._annopath = 'Flickr30kEntities/Annotations/{}.xml'
        self._imgpath = 'flickr30k_images/{}.jpg'
        self.ids = list()
        # rootpath = '/scratch/arka/Ark_git_files/flickr30k'
        for line in open(osp.join(str(self.root), 'original_qrc_lists', f'{self.name}_{dset}.lst')):
            self.ids.append(line.strip())

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self.ficd[str(img_id)]

        img = cv2.imread(str(self.root / self._imgpath.format(img_id)))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(
                img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


if __name__ == '__main__':
    dataset = FLICK_Detection(root='/scratch/arka/Ark_git_files/flickr30k')
