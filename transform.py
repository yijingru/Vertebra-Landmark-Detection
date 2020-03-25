
import numpy as np
from numpy import random
import cv2


def rescale_pts(pts, down_ratio):
    return np.asarray(pts, np.float32)/float(down_ratio)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts

class ConvertImgFloat(object):
    def __call__(self, img, pts):
        return img.astype(np.float32), pts.astype(np.float32)

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, pts):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, pts


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, pts):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, pts

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, pts):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, pts


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, pts):
        img, pts = self.rb(img, pts)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img, pts = distort(img, pts)
        img, pts = self.rln(img, pts)
        return img, pts


class Expand(object):
    def __init__(self, max_scale = 1.5, mean = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.max_scale = max_scale

    def __call__(self, img, pts):
        if random.randint(2):
            return img, pts
        h,w,c = img.shape
        ratio = random.uniform(1,self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        if np.max(pts[:,0])+int(x1)>w-1 or np.max(pts[:,1])+int(y1)>h-1:  # keep all the pts
            return img, pts
        else:
            expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
            expand_img[:,:,:] = self.mean
            expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
            pts[:, 0] += int(x1)
            pts[:, 1] += int(y1)
            return expand_img, pts


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win = 0.9):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            # (0.1, None),
            # (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, pts):
        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, pts
            for _ in range(50):
                current_img = img
                current_pts = pts
                w = random.uniform(self.min_win*width, width)
                h = random.uniform(self.min_win*height, height)
                if h/w<self.ratio[0] or h/w>self.ratio[1]:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                current_pts[:, 0] -= rect[1]
                current_pts[:, 1] -= rect[0]
                pts_new = []
                for pt in current_pts:
                    if any(pt)<0 or pt[0]>current_img.shape[1]-1 or pt[1]>current_img.shape[0]-1:
                        continue
                    else:
                        pts_new.append(pt)

                return current_img, np.asarray(pts_new, np.float32)

class RandomMirror_w(object):
    def __call__(self, img, pts):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1,:]
            pts[:,0] = w-pts[:,0]
        return img, pts

class RandomMirror_h(object):
    def __call__(self, img, pts):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:,:]
            pts[:,1] = h-pts[:,1]
        return img, pts


class Resize(object):
    def __init__(self, h, w):
        self.dsize = (w,h)

    def __call__(self, img, pts):
        h,w,c = img.shape
        pts[:, 0] = pts[:, 0]/w*self.dsize[0]
        pts[:, 1] = pts[:, 1]/h*self.dsize[1]
        img = cv2.resize(img, dsize=self.dsize)
        return img, np.asarray(pts)