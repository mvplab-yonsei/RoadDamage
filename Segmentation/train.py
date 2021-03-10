from dataloaders import make_data_loader
from utils.metrics import Evaluator
import torch
import torch.nn as nn
import torchvision as tv
import segmentation_models_pytorch as smp
from PIL import Image


class Trainer(object):
    def __init__(self):
        self.train_loader, self.val_loader = make_data_loader(batch=32, crop_size=480)

        # model define
        model = smp.demodel = smp.PAN(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=2)
        self.model = model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.evaluator = Evaluator(2)

    def training(self):
        best_miou = 0
        for epoch in range(3000):

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
                    print('[Epoch: {}] Train loss: {:.5f}'.format(epoch, (train_loss / (i + 1))))

            # val
            self.model.eval()
            self.evaluator.reset()
            for i, sample in enumerate(self.val_loader):
                # image: (B,3,H,W)
                # mask: (B,H,W)
                image, mask = sample['image'], sample['mask']
                image = image.cuda()
                mask[mask != 0] = 1
                mask = mask.cuda().squeeze(1).long()

                # run
                with torch.no_grad():
                    output = self.model(image)

                # pred: (B,H,W)
                pred = torch.argmax(output, dim=1)

                # add batch sample into evaluator
                self.evaluator.add_batch(mask.cpu().numpy(), pred.cpu().numpy())

                # save output
                if i % 50 == 49:
                    tv.utils.save_image(pred.float().unsqueeze(0), 'output/{}_pred_101.png'.format(i))
                    tv.utils.save_image(mask.float().unsqueeze(0), 'output/{}_gt_101.png'.format(i))
                    tv.utils.save_image(image[0] * 0.2 + 0.5, 'output/{}_image.png'.format(i))

            # testing
            Acc = self.evaluator.Pixel_Accuracy()
            miou = self.evaluator.Mean_Intersection_over_Union()
            if miou > best_miou:
                best_miou = miou
            print('[Epoch: {}] Validation:'.format(epoch))
            print('Acc:{:5f}, mIoU:{:5f}, best mIoU:{:5f}\n'.format(Acc, miou, best_miou))

            # save model
            if epoch % 50 == 0:
                state = self.model.state_dict()
                torch.save(state, './trained_model/{}_101.tar'.format(epoch))


def main():
    torch.cuda.set_device(0)

    for i in range(100):
        img = tv.transforms.ToTensor()(Image.open('./results/199_{}_image.png'.format(i))).cuda()
        pred = tv.transforms.ToTensor()(Image.open('./results/199_{}_pred_34.png'.format(i))).cuda()
        kernel = torch.ones(1, 1, 41, 41).cuda()
        pred = torch.clamp(torch.nn.functional.conv2d(pred.unsqueeze(1), kernel, padding=(20, 20)), 0, 1).squeeze(1)
        pred[1:] = 0
        tv.utils.save_image(img + pred * 0.5, './guide_dil_20/{}.png'.format(i))
    exit()
    trainer = Trainer()
    trainer.training()


if __name__ == "__main__":
   main()
