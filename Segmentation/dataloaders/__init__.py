from dataloaders import road
from torch.utils.data import DataLoader


def make_data_loader(batch, crop_size):
    train_set = road.ROADSegmentation(split='train', crop_size=crop_size)
    val_set = road.ROADSegmentation(split='val', crop_size=crop_size)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    return train_loader, val_loader


