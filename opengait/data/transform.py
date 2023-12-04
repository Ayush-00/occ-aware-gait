import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.divsor


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std


class RandomOccludeTransform():
    def __init__(self, occlude_bounds = [0.0, 0.0], img_w=64, disvor=255.0):
        self.img_w = img_w
        self.disvor = disvor
        #self.gray_prob = gray_prob
        self.occlude_bounds = occlude_bounds
        #self.gray_aug = RandomGrayScale(prob=self.gray_prob)
        return
        
    def __call__(self, x):
        
        x = self.random_occlude(x)
        
        #rgb = x[:,:,:,:3]
        mask = x
        
        #rgb = self.gray_aug(rgb)        #If prob is 0, then same video is returned.
        
        if self.disvor == 'max':
            # if np.isnan(np.max(rgb)) or np.max(rgb) == 0:
            #     rgb_div = 255.0
            # else: 
            #     rgb_div = np.max(rgb)
            # rgb = rgb / rgb_div
            mask = mask / np.max(mask)
        else:
            #rgb = rgb / self.disvor
            mask = mask / self.disvor
        #x = np.concatenate((rgb, mask), axis=-1)
        
        x = mask
        return x
    
    def random_occlude(self, x):
        # x: f,h,w,c -> Single video
        
        """ Randomly choose one of the 8 occlusion types, and occlude the video """
    
        ######### OCCLUSION KEY ################
        # 0 - No occlusion
        # 1 - top right occluded
        # 2 - top left occluded
        # 3 - bottom left occluded
        # 4 - bottom right occluded
        # 5 - bottom occluded
        # 6 - top occluded
        # 7 - left occluded
        # 8 - right occluded
        
        # NOTE: the vertical height of occlusion can vary everywhere, but
        # the position of occlusion horizontally is fixed to be in the middle - 
        # the amount of person on left is always the same as on the right 
        ##########################################
        
        
        f,h,w = x.shape
        occ_typ = random.randint(0, 8)
        
        if self.occlude_bounds[0] == 0.0 and self.occlude_bounds[1] == 0.0:
            occ_typ = 0
            return x
        
        if occ_typ == 0:
            pass    # No occlusion
        elif occ_typ in [1,2,3,4]:
            occ_height = random.random()*(self.occlude_bounds[1]-self.occlude_bounds[0]) + self.occlude_bounds[0]
            
            if occ_typ == 1:
                x[:,:int(h*occ_height), w//2:] = 0
            elif occ_typ == 2:
                x[:int(h*occ_height), :w//2] = 0
            elif occ_typ == 3:  
                x[int(h*(1-occ_height)):, :w//2] = 0
            elif occ_typ == 4:
                x[int(h*(1-occ_height)):, w//2:] = 0
            
            
            
            
        elif occ_typ in [5,6]:
            # Occluded height can be ANYTHING from within a range, cut the video and resize it to original (h,w) while maintaining ratio!
            occ_height = random.random()*(self.occlude_bounds[1]-self.occlude_bounds[0]) + self.occlude_bounds[0]
            
            if occ_typ == 5:
                crop_x = x[:,:int((1-occ_height)*h), :]
            elif occ_typ == 6:
                crop_x = x[:,int(occ_height*h):, :]
            
            center_w = w//2
            crop_width_x = crop_x[:,:,int(center_w-((1-occ_height)*w/2)):int(center_w+((1-occ_height)*w/2))]
            
            # interpolations = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA]
            # chosen_interpol = interpolations[torch.randint(low=0, high=len(interpolations), size=(1,)).item()]
            
            x = np.array([cv2.resize(frame, dsize=(w,h)) for frame in crop_width_x])
            
        elif occ_typ == 7:
            x[:,:,:w//2] = 0
        elif occ_typ == 8:
            x[:,:,w//2:] = 0

        #print(x.shape, x.dtype)
        return x

# **************** Data Agumentation ****************


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]


class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for _ in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]

                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1+h, y1:y1+w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...])
                   for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, h, w = seq.shape
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            pts2 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            return seq
        
# ******************************************

def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"
