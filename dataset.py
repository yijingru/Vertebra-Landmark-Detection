import os
import torch.utils.data as data
import pre_proc
import cv2
from scipy.io import loadmat
import numpy as np


def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 68
        self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))

    def load_image(self, index):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        return image

    def load_gt_pts(self, annopath):
        pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        pts = rearrange_pts(pts)
        return pts

    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'labels', self.phase, img_id+'.mat')

    def load_annotation(self, index):
        img_id = self.img_ids[index]
        annoFolder = self.load_annoFolder(img_id)
        pts = self.load_gt_pts(annoFolder)
        return pts

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = self.load_image(index)
        if self.phase == 'test':
            images = pre_proc.processing_test(image=image, input_h=self.input_h, input_w=self.input_w)
            return {'images': images, 'img_id': img_id}
        else:
            aug_label = False
            if self.phase == 'train':
                aug_label = True
            pts = self.load_annotation(index)   # num_obj x h x w
            out_image, pts_2 = pre_proc.processing_train(image=image,
                                                         pts=pts,
                                                         image_h=self.input_h,
                                                         image_w=self.input_w,
                                                         down_ratio=self.down_ratio,
                                                         aug_label=aug_label,
                                                         img_id=img_id)

            data_dict = pre_proc.generate_ground_truth(image=out_image,
                                                       pts_2=pts_2,
                                                       image_h=self.input_h//self.down_ratio,
                                                       image_w=self.input_w//self.down_ratio,
                                                       img_id=img_id)
            return data_dict

    def __len__(self):
        return len(self.img_ids)
