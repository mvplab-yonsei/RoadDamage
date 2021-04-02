from dataloaders import make_data_loader
import torch
import torch.nn as nn
import torchvision as tv
import segmentation_models_pytorch as smp
import os
from utils.logger import setup_logger
import logging
from datetime import datetime
import psutil
import platform
import utils.cpuinfo as cpuinfo
from pynvml import *
import xml.etree.ElementTree as ET
import argparse


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def write_info(logger, is_started):
    if is_started:
        uname = platform.uname()
        logger.info(f"Test Command: 'python test.py'")
        logger.info("======================================Boot Time======================================")
        logger.info(f"Boot Time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
        logger.info("==================================System Information==================================")
        logger.info(f"System: {uname.system}\nNode Name: {uname.node}\nRelease: {uname.release}\nVersion: {uname.version}\nMachine: {uname.machine}\nProcessor: {uname.processor}")
        svmem = psutil.virtual_memory()
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        logger.info(f"CPU: {cpuinfo.cpu.info[0]['model name']}\nMemory total: {get_size(svmem.total)}\nGPU: {nvmlDeviceGetName(handle)}")
        logger.info(f"Pytorch Version: {torch.__version__}")
    else:
        logger.info("======================================Boot Time======================================")
        logger.info(f"Boot Time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")



def Logger_System(xml_dir, log_dir, output, paths, threshold):
    xml_file = os.path.join(xml_dir, paths.split('/')[-1].replace('jpg', 'xml'))
    log_file = os.path.join(log_dir, paths.split('/')[-1].replace('jpg', 'txt'))

    if output > threshold:
        f = open(log_file, 'a+')
        if os.path.isfile(log_file):
            if os.path.isfile(xml_file):
                ann_tree = ET.parse(xml_file)
                ann_root = ann_tree.getroot()
                env = ann_root.find('environment').find('Gps_sensor')
                f.write("[GPS information]:\n")
                f.write(env.findtext('Altitude') + '\n')
                f.write(env.findtext('Longitude') + '\n')
                f.write(env.findtext('Status') + '\n')

        f.write("\n[Crack]:\n")
        f.write("True" + '\n')
        f.close()


class Tester(object):
    def __init__(self, use_logger, epoch):
        self.val_loader = make_data_loader()
        self.model = smp.PAN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2).cuda()
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load('./trained_model/{}_34.tar'.format(epoch)))
        self.model.eval()
        self.use_logger = use_logger
        if not os.path.isdir('./output'):
            os.mkdir('./output')
            os.mkdir('./output/img')
            os.mkdir('./output/pred')
            os.mkdir('./output/mask')

    def testing(self):
        total_acc = 0
        total_miou = 0

        logger = logging.getLogger('Test')

        for i, sample in enumerate(self.val_loader):
            # image: (1,3,H,W)
            # mask: (1,1,H,W)
            image, mask, name = sample['image'], sample['mask'], sample['name']
            image = image.cuda()
            mask[mask != 0] = 1
            mask = mask.cuda().long()

            # run
            with torch.no_grad():
                output = self.model(image)

            # pred: (1,1,H,W), 0 or 1
            FG_pred = torch.argmax(output, dim=1).unsqueeze(0).float()
            BG_pred = 1 - FG_pred

            # Calculate Pixels
            if self.use_logger:
                xml_dir = '../../database/data_0310/xmls'
                log_dir = '../Logger'
                paths = './testset/images/'
                Crack_pixels = torch.count_nonzero(FG_pred)
                threshold = 2 ** 10
                Logger_System(xml_dir, log_dir, Crack_pixels, paths + name[0], threshold)

            # dilation kernel for pixel margin
            margin = 2
            kernel = torch.ones(1, 1, 2 * margin + 1, 2 * margin + 1).cuda()
            dil_FG_pred = torch.clamp(torch.nn.functional.conv2d(FG_pred, kernel, padding=(margin, margin)), 0, 1)
            dil_BG_pred = torch.clamp(torch.nn.functional.conv2d(BG_pred, kernel, padding=(margin, margin)), 0, 1)

            # evaluate
            total_correct_pixels = torch.sum(FG_pred * mask + BG_pred * (1 - mask))
            total_pixels = mask.size(-1) * mask.size(-2)
            pixel_acc = total_correct_pixels / total_pixels
            total_acc += pixel_acc
            FG_total = torch.sum((FG_pred + mask).clamp(max=1))
            FG_correct = torch.sum(dil_FG_pred * mask)
            BG_total = torch.sum((BG_pred + 1 - mask).clamp(max=1))
            BG_correct = torch.sum(dil_BG_pred * (1 - mask))
            miou = (FG_correct / FG_total + BG_correct / BG_total) / 2
            total_miou += miou

            logger.info('[{}] {} [Pixal Acc.:{:5f}] [mIoU:{:5f}]'.format(str(i+1), name[0], pixel_acc, miou))
            logger.info('(total # of pixels:{}, total # of correct pixels:{}, FG Union:{}, FG Intersection:{}, BG Union:{}, BG Intersection:{})'.
                        format(int(total_pixels), int(total_correct_pixels), int(FG_total), int(FG_correct), int(BG_total), int(BG_correct)))

            # # save results
            # tv.utils.save_image((image * 0.2) + 0.5, './output/img/{}.png'.format(i))
            # tv.utils.save_image(FG_pred, './output/pred/{}.png'.format(i))
            # tv.utils.save_image(mask.float(), './output/mask/{}.png'.format(i))

        # testing
        total_acc /= (i + 1)
        total_miou /= (i + 1)
        logger.info('[Final] [Pixel accuracy:{:5f}, mIoU:{:5f}]\n'.format(total_acc, total_miou))


def main():
    torch.cuda.set_device(0)
    logger = setup_logger("Test", './', 0)
    write_info(logger, True)
    parser = argparse.ArgumentParser(description="Road Segmentation Test")
    parser.add_argument("--use_logger_system", action='store_true')
    args = parser.parse_args()
    trainer = Tester(args.use_logger_system, epoch=25)
    trainer.testing()
    write_info(logger, False)


if __name__ == "__main__":
    main()
