import os

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils import data
import torch.nn as nn
import random


np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.8
NUM_PTS = 971
CROP_SIZE = 224
CROP_SIZE_H = 166
CROP_SIZE_W = 128
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        # self.size = torch.tensor(size, dtype=torch.float)
        self.size = np.asarray(size, dtype=np.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample

class CropRectangle(object):
    def __init__(self, size=(CROP_SIZE_H, CROP_SIZE_W), elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape

        margin_h = (h - self.size[0]) // 2
        margin_w = (w - self.size[1]) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size[0], margin_w:margin_w + self.size[1]]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            # print('Crop')
            # print(landmarks[:5])
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            # print(landmarks[:5])
            # print('End crop')
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


# Преобразование для удаление черной рамки вокруг изображения
class CropFrame(object):
    def __init__(self, limit=9, elem_name='image'):
        self.limit = limit
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape

        brightness = np.mean(img, axis=2)
        top = 0
        while (top < h) and np.max(brightness[top, :]) <= self.limit:
            top += 1

        bottom = h - 1
        while (bottom > 0) and np.max(brightness[bottom, :]) <= self.limit:
            bottom += -1

        left = 0
        while (left < w) and np.max(brightness[:, left]) <= self.limit:
            left += 1

        right = w - 1
        while (right > 0) and np.max(brightness[:, right]) <= self.limit:
            right += -1

        sample[self.elem_name] = img[top:bottom+1, left:right+1, :]

        sample["crop_top"] = top
        sample["crop_left"] = left

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((left, top), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


# Преобразование - отражение от вертикальной оси
class FlipHorizontal(object):
    def __init__(self, p=0.5, elem_name='image'):
        self.elem_name = elem_name
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            w = sample[self.elem_name].shape[1]

            sample[self.elem_name] = cv2.flip(sample[self.elem_name], 1)

            if 'landmarks' in sample:
                landmarks = sample['landmarks'].reshape(-1, 2)
                landmarks[:, 0] = torch.tensor(w, dtype=landmarks.dtype) - landmarks[:, 0]
                final_landmarks = landmarks.clone()
                # Низ овала лица
                final_landmarks[:64] = landmarks[64:128]
                final_landmarks[64:128] = landmarks[:64]

                # Верх овала лица
                final_landmarks[128:273] = landmarks[128:273].flip(0)

                # Брови
                final_landmarks[273:337] = landmarks[337:401]
                final_landmarks[337:401] = landmarks[273:337]

                # Нос
                final_landmarks[401:464] = landmarks[464:527]
                final_landmarks[464:527] = landmarks[401:464]

                # Глаза
                final_landmarks[587:714] = landmarks[714:841]
                final_landmarks[714:841] = landmarks[587:714]

                # Верхняя губа верх
                final_landmarks[841:873] = landmarks[841:873].flip(0)
                final_landmarks[873:905] = landmarks[873:905].flip(0)

                # Верхняя нуба низ
                final_landmarks[905:937] = landmarks[905:937].flip(0)
                final_landmarks[937:969] = landmarks[937:969].flip(0)

                # Глаза
                final_landmarks[969:972] = landmarks[969:972].flip(0)

                sample['landmarks'] = final_landmarks.reshape(-1)

        return sample


# Преобразование - изменение яркость/контрастности
class ChangeBrightnessContrast(object):
    def __init__(self, alpha_std=1, beta_std=0, elem_name='image'):
        self.elem_name = elem_name
        self.alpha_std = alpha_std
        self.beta_std = beta_std

    def __call__(self, sample):
        alpha = np.random.normal(1.0, self.alpha_std)
        beta = np.random.normal(0.0, self.beta_std)
        sample[self.elem_name] = cv2.convertScaleAbs(sample[self.elem_name],
                                                     alpha=alpha,
                                                     beta=beta)
        return sample


# Преобразование - поворот вокруг центра
class Rotator(object):
    def __init__(self, max_angle=0, elem_name='image'):
        self.elem_name = elem_name
        self.max_angle = max_angle

    def __call__(self, sample):
        angle = random.uniform(-self.max_angle, self.max_angle)
        center = (sample[self.elem_name].shape[0]//2,
                  sample[self.elem_name].shape[1] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
        sample[self.elem_name] = cv2.warpAffine(
            sample[self.elem_name],
            rot_mat,
            (sample[self.elem_name].shape[1],
                sample[self.elem_name].shape[0]
             )
        )
        if 'landmarks' in sample:
            landmarks = sample['landmarks'].float()
            landmarks = self.rotate_landmarks(center, landmarks, angle)
            sample['landmarks'] = landmarks.reshape(-1)
        return sample

    def rotate_landmarks(self, center, points, angle):
        x_c, y_c = center
        angle_rad = -angle * np.pi / 180
        points = points.reshape(-1, 2)
        landmarks = points.clone().detach()
        landmarks[:, 0] = x_c + np.cos(angle_rad) * (points[:, 0] - x_c) - np.sin(angle_rad) * (points[:, 1] - y_c)
        landmarks[:, 1] = y_c + np.sin(angle_rad) * (points[:, 0] - x_c) + np.cos(angle_rad) * (points[:, 1] - y_c)
        return landmarks


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train"):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split != "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp), total=num_lines + 1):
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(TRAIN_SIZE * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(TRAIN_SIZE * num_lines):
                    continue  # has not reached start of val part of data
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int).reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))