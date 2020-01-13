import cv2
import torch
import numpy as np
import pandas as pd
from mask_deal import *
from torch.utils.data import DataLoader, Dataset

class MYdata(Dataset):
    def __init__(self, csv_path, size=(768, 256), mode='train'):
        super(MYdata, self).__init__()
        total_list = pd.read_csv(csv_path)
        self.mode = mode
        self.size = size
        total_list_shuffle = total_list.sample(frac=1.0)  # 全部打乱
        cut_idx = int(round(0.1 * total_list_shuffle.shape[0]))
        val_list, train_list = total_list_shuffle.iloc[:cut_idx], total_list_shuffle.iloc[cut_idx:]

        if self.mode == 'train':
            self.data_list = train_list
        else:
            self.data_list = val_list

    def crop_resize_data(self, image, offset=690,interpolation=cv2.INTER_LINEAR):
        roi_image = image[offset:, :]
        image_size=self.size
        out = cv2.resize(roi_image, (image_size[0], image_size[1]), interpolation=interpolation)
        return out

    def get_img(self, image_dir):
        ori_image = cv2.imread(image_dir, 1)
        image = self.crop_resize_data(ori_image)
        image = image.transpose((2,0,1))
        return image.astype('float32')/255.0

    def get_label(self, label_dir):
        ori_mask = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        mask = self.crop_resize_data(ori_mask,interpolation=cv2.INTER_NEAREST)
        encode_mask = encode_labels(mask)
        return encode_mask.astype('int64')

    def __getitem__(self, index):
        image_dir = np.array(self.data_list['image'])[index]
        label_dir = np.array(self.data_list['label'])[index]
        img = self.get_img(image_dir)
        
        label = self.get_label(label_dir)
 
#        label_show = label*30
#        img_show = img.transpose((1,2,0))
#        cv2.imwrite('show.jpg', img_show*255.0)
#        cv2.imwrite('show_label.jpg', label_show)
#        raise RuntimeError      
        return img, label

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    pass
