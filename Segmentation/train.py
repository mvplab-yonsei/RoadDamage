from dataloaders import make_data_loader
import torch
import torch.nn as nn
import torchvision as tv
import segmentation_models_pytorch as smp
from datetime import datetime
import numpy as np
import random
from utils.logger import setup_logger
import logging
import cudnn


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


class Trainer(object):
    def __init__(self):
        self.backbone = 'resnet34'
        self.lr = 1e-4
        self.batch = 64
        self.crop_size = 384
        logger = logging.getLogger('train')

        # print settings
        logger.info('\nTRAINING SETTINGS')
        logger.info('###########################')
        logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info('backbone:{}, lr:{}, batch:{}, crop size:{}'.format(self.backbone, self.lr, self.batch, self.crop_size))
        logger.info('###########################\n')

        # model define
        self.train_loader, self.test_loader = make_data_loader(batch=self.batch, crop_size=self.crop_size)
        model = smp.PAN(encoder_name=self.backbone, encoder_weights='imagenet', in_channels=3, classes=2)
        self.model = nn.DataParallel(model.cuda())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 0

    def training(self):
        best_miou = 0
        logger = logging.getLogger('train')

        for epoch in range(self.epoch, self.epoch + 100):
            # train
            train_loss = 0.0
            self.model.train()
            for i, sample in enumerate(self.train_loader):
                # image: (B,3,H,W)
                # mask: (B,1,H,W)
                image, mask = sample['image'], sample['mask']
                image = image.cuda()
                mask[mask != 0] = 1
                mask = mask.cuda().squeeze(1).long()

                # output: (B,2,H,W)
                self.optimizer.zero_grad()
                output = self.model(image)

                # loss
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                if i % 50 == 49:
                    logger.info('[Epoch: {}] Train loss: {:.5f}'.format(epoch, (train_loss / (i + 1))))

            # test
            self.model.eval()
            total_acc = 0
            total_miou = 0
            for i, sample in enumerate(self.test_loader):
                # image: (1,3,H,W)
                # mask: (1,1,H,W)
                image, mask = sample['image'], sample['mask']
                image = image.cuda()
                mask[mask != 0] = 1
                mask = mask.cuda().long()

                # run
                with torch.no_grad():
                    output = self.model(image)

                # pred: (1,1,H,W), 0 or 1
                FG_pred = torch.argmax(output, dim=1).unsqueeze(0).float()
                BG_pred = 1 - FG_pred

                # dilation kernel for pixel margin
                margin = 2
                kernel = torch.ones(1, 1, 2 * margin + 1, 2 * margin + 1).cuda()
                dil_FG_pred = torch.clamp(torch.nn.functional.conv2d(FG_pred, kernel, padding=(margin, margin)), 0, 1)
                dil_BG_pred = torch.clamp(torch.nn.functional.conv2d(BG_pred, kernel, padding=(margin, margin)), 0, 1)

                # evaluate
                pixel_acc = torch.sum(FG_pred * mask + BG_pred * (1 - mask)) / mask.size(-1) / mask.size(-2)
                total_acc += pixel_acc
                FG_total = torch.sum((FG_pred + mask).clamp(max=1))
                FG_correct = torch.sum(dil_FG_pred * mask)
                BG_total = torch.sum((BG_pred + 1 - mask).clamp(max=1))
                BG_correct = torch.sum(dil_BG_pred * (1 - mask))
                miou = (FG_correct / FG_total + BG_correct / BG_total) / 2
                total_miou += miou

            # testing
            total_acc /= (i + 1)
            total_miou /= (i + 1)
            if total_miou > best_miou:
                best_miou = total_miou
            logger.info('[Epoch: {}] Validation:'.format(epoch))
            logger.info('Acc:{:5f}, mIoU:{:5f}, best mIoU:{:5f}\n'.format(total_acc, total_miou, best_miou))

            # save model
            state = self.model.state_dict()
            torch.save(state, './trained_model/{}_34.tar'.format(epoch))


def main():
    seed = 0
    init_seeds(seed)
    init_torch_seeds(seed)
    torch.cuda.set_device(0)
    logger = setup_logger('train', './', 0)
    trainer = Trainer()
    trainer.training()


if __name__ == "__main__":
    main()
