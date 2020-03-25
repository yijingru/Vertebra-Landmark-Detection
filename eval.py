import torch
import numpy as np
from models import spinal_net
import decoder
import os
from dataset import BaseDataset
import time
import cobb_evaluate

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,  # cen, tl, tr, bl, br
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model


    def eval(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        landmark_dist = []
        pr_cobb_angles = []
        gt_cobb_angles = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']
            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            x_index = range(0,10,2)
            y_index = range(1,10,2)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0[:, x_index] = pts0[:, x_index]/args.input_w*w
            pts0[:, y_index] = pts0[:, y_index]/args.input_h*h
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                    landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image))
            gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_landmarks, ori_image))

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)

        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = np.sum(out_abs, axis=1)
        term2 = np.sum(out_add, axis=1)

        SMAPE = np.mean(term1 / term2 * 100)

        print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))
        print('SMAPE is {}'.format(SMAPE))

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))


    def SMAPE_single_angle(self, gt_cobb_angles, pr_cobb_angles):
        out_abs = abs(gt_cobb_angles - pr_cobb_angles)
        out_add = gt_cobb_angles + pr_cobb_angles

        term1 = out_abs
        term2 = out_add

        term2[term2==0] += 1e-5

        SMAPE = np.mean(term1 / term2 * 100)
        return SMAPE

    def eval_three_angles(self, args, save):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        landmark_dist = []
        pr_cobb_angles = []
        gt_cobb_angles = []
        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']
            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio
            x_index = range(0,10,2)
            y_index = range(1,10,2)
            ori_image = dsets.load_image(dsets.img_ids.index(img_id)).copy()
            h,w,c = ori_image.shape
            pts0[:, x_index] = pts0[:, x_index]/args.input_w*w
            pts0[:, y_index] = pts0[:, y_index]/args.input_h*h
            # sort the y axis
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)   #[68, 2]

            end_time = time.time()
            total_time.append(end_time-begin_time)

            gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id))
            for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                    landmark_dist.append(np.sqrt((pr_pt[0]-gt_pt[0])**2+(pr_pt[1]-gt_pt[1])**2))

            pr_cobb_angles.append(cobb_evaluate.cobb_angle_calc(pr_landmarks, ori_image))
            gt_cobb_angles.append(cobb_evaluate.cobb_angle_calc(gt_landmarks, ori_image))

        pr_cobb_angles = np.asarray(pr_cobb_angles, np.float32)
        gt_cobb_angles = np.asarray(gt_cobb_angles, np.float32)


        print('SMAPE1 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,0], pr_cobb_angles[:,0])))
        print('SMAPE2 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,1], pr_cobb_angles[:,1])))
        print('SMAPE3 is {}'.format(self.SMAPE_single_angle(gt_cobb_angles[:,2], pr_cobb_angles[:,2])))

        print('mse of landmarkds is {}'.format(np.mean(landmark_dist)))

        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1./np.mean(total_time)))

