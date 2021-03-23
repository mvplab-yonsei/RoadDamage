import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision as tv
import random
import torchvision.transforms.functional as TF


class ROADSegmentation(Dataset):
    NUM_CLASSES = 2

    def __init__(self, base_dir='../DB/Road/', split='train', crop_size=400, mode=None):
        super().__init__()
        self.base_dir = base_dir + split + '/'
        self.img_dir = os.path.join(self.base_dir, 'images')
        self.anno_dir = os.path.join(self.base_dir, 'anno2')
        self.split = split
        self.crop_size = crop_size
        self.mode = mode
        self.images = []
        self.masks = []
        self.test_images = []

        for ii, line in enumerate(os.listdir(self.img_dir)):
            img = os.path.join(self.img_dir, line)
            mask = os.path.join(self.anno_dir, line.split('.')[0] + '.png')
            self.images.append(img)
            self.masks.append(mask)
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {}
        if self.split == 'train':
            img, mask = self.make_img_gt_point_pair(index)
            img, mask = self.transform(img, mask)
            sample['image'] = self.val_transform(img)
            sample['mask'] = self.val_label_transform(mask)
            return sample
        elif self.split == 'val':
            img, mask = self.make_img_gt_point_pair(index)
            sample['image'] = self.val_transform(img)
            sample['mask'] = self.val_label_transform(mask)
            return sample

    def make_img_gt_point_pair(self, index):
        img = Image.open(self.images[index])
        mask = Image.open(self.masks[index]).convert('P')
        return img, mask

    def transform(self, image, mask):

        # balanced random crop
        for cnt in range(10):
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            temp_mask = TF.crop(mask, i, j, h, w)
            if len(transforms.ToTensor()(mask).unique()) > 1:
                image = TF.crop(image, i, j, h, w)
                mask = temp_mask
                break
            if cnt == 9:
                image = TF.crop(image, i, j, h, w)
                mask = temp_mask

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def val_transform(self, sample):
        tran = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        return tran(sample)

    def val_label_transform(self, sample):
        tran = tv.transforms.Compose([
            tv.transforms.ToTensor()])
        return tran(sample)



