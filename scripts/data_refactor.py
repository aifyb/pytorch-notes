# -*- coding: utf-8 -*-
'''
@File    :   data_refactor.py
@Time    :   2023/06/26 22:36:19
@Author  :   feng yongbing
'''

import os
import shutil

# 1. 原始数据在上级目录 ants_bees 文件夹下，包含 train和val两个文件夹，其下文件夹为类名，文件夹内为图片数据；
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'ants_bees')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# 2.检测当前目录下是否存在refactor文件夹，存在则删除，不存在则创建；
refactor_dir = os.path.join(os.getcwd(), 'refactor')
if os.path.exists(refactor_dir):
    shutil.rmtree(refactor_dir)
os.mkdir(refactor_dir)

# 3.在其下创建train和val两个文件夹，并分别包含image和label两个文件夹；
train_image_dir = os.path.join(refactor_dir, 'train', 'image')
train_label_dir = os.path.join(refactor_dir, 'train', 'label')
val_image_dir = os.path.join(refactor_dir, 'val', 'image')
val_label_dir = os.path.join(refactor_dir, 'val', 'label')
os.makedirs(train_image_dir)
os.makedirs(train_label_dir)
os.makedirs(val_image_dir)
os.makedirs(val_label_dir)

# # 4. 复制原始图片至对应文件夹下，按照类别在label里面创建和图片的对应txt文件
for mode in ['train', 'val']:
    mode_dir = train_dir if mode == 'train' else val_dir
    for class_name in os.listdir(mode_dir):
        class_dir = os.path.join(mode_dir, class_name)
        image_dir = train_image_dir if mode == 'train' else val_image_dir
        label_dir = train_label_dir if mode == 'train' else val_label_dir
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            shutil.copy(image_path, os.path.join(image_dir, image_name))
            with open(os.path.join(label_dir, image_name[:-4] + '.txt'), 'w') as f:
                f.write(class_name)
