from dataloaders import make_data_loader
import torch
import torch.nn as nn
import torchvision as tv
import segmentation_models_pytorch as smp
from utils.logger import setup_logger
import logging
from datetime import datetime
import psutil
import platform
import utils.cpuinfo as cpuinfo
from pynvml import *
import xml.etree.ElementTree as ET


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


def write_info(logger):
    uname = platform.uname()
    logger.info("======================================Boot Time======================================")
    logger.info(f"Boot Time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    logger.info("==================================System Information==================================")
    logger.info(f"System: {uname.system}\nNode Name: {uname.node}\nRelease: {uname.release}\nVersion: {uname.version}\nMachine: {uname.machine}\nProcessor: {uname.processor}")
    svmem = psutil.virtual_memory()
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    logger.info(f"CPU: {cpuinfo.cpu.info[0]['model name']}\nMemory total: {get_size(svmem.total)}\nGPU: {nvmlDeviceGetName(handle)}")


def Logger_System(xml_dir, log_dir, output, paths, threshold):
    xml_file = os.path.join(xml_dir, paths.split('/')[-1].replace('jpg', 'xml'))
    log_file = os.path.join(log_dir, paths.split('/')[-1].replace('jpg', 'txt'))
    print(xml_file)
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
    def __init__(self, epoch):
        # _, self.val_loader = make_data_loader()
        self.val_loader = make_data_loader()
        self.model = smp.PAN(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2).cuda()
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load('/data1/submit_code/Test_code_segmentation/trained_model/{}_34.tar'.format(epoch), map_location='cuda:0'))
        self.model.eval()
        self.xml_dir = '/data1/database/data_0310/xmls'
        self.log_dir = '/data1/submit_code/Logger'
        self.paths = '/data1/submit_code/Test_code_segmentation/testset/images/'
        self.threshold = 2048
        self.use_logger = True
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
                Crack_pixels = torch.count_nonzero(FG_pred)
                Logger_System(self.xml_dir, self.log_dir, Crack_pixels, self.paths + name[0], self.threshold)

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

            logger.info('[{}] {} [Pixel accuracy:{:5f}, mIou:{:5f}]'.format(str(i+1), name[0], pixel_acc, miou))

            # save results
            tv.utils.save_image((image * 0.2) + 0.5, './output/img/{}'.format(name[0]))
            tv.utils.save_image(FG_pred, './output/pred/{}'.format(name[0]))
            tv.utils.save_image(mask.float(), './output/mask/{}'.format(name[0]))

        # testing
        total_acc /= (i + 1)
        total_miou /= (i + 1)

        logger.info('[Final] [Pixel accuracy:{:5f}, mIoU:{:5f}]\n'.format(total_acc, total_miou))


def main():
    torch.cuda.set_device(0)

    logger = setup_logger("Test", './', 0)
    write_info(logger)
    trainer = Tester(epoch=40)
    trainer.testing()


if __name__ == "__main__":
    main()
