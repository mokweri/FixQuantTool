import io
import os
from zipfile import ZipFile
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class ImageNetDataset(Dataset):
    def __init__(self, dataroot: str, train: bool = True, transform=None):
        self.zf = ZipFile(
            os.path.join(
                dataroot,
                f"{'train' if train else 'val'}_blurred.zip",
            )
        )
        self.imglist: list[str] = [
            path for path in self.zf.namelist()
            if path.endswith(".jpg")
        ]

        # Images are structured in directories based on class
        with open(os.path.join(dataroot, "map_clsloc.txt")) as f:
            def parse_row(row: str) -> tuple[str, int]:
                classname, classnum, _ = row.split()
                return classname, (int(classnum) - 1)

            self.classes: dict[str, int] = dict(parse_row(row) for row in f)

        self.transform = transform

    def get_label(self, path: str) -> int:
        if not path.endswith(".jpg"):
            raise ValueError(f"Expected path to image, got {path}")
        classname: str = path.split("/")[-2]
        return self.classes[classname]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        imgpath = self.imglist[idx]
        img = Image.open(io.BytesIO(self.zf.read(imgpath)))
        label = self.get_label(imgpath)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataloaders(dataroot, batch_size, num_workers):
    '''Initializes and returns a dataloader.'''

    # Init transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Init dataset
    train_dataset = ImageNetDataset(dataroot, train=True, transform=train_transforms)
    val_dataset = ImageNetDataset(dataroot, train=False, transform=val_transforms)

    # Init dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=1, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    dataroot = '/mimer/NOBACKUP/Datasets/ImageNet/Face-blurred_ILSVRC2012-2017'
    # /mimer/NOBACKUP/groups/naiss2024-22-1352/Obed_Work/ILSVRC2012

    train_loader, val_loader = get_dataloaders(dataroot=dataroot, batch_size=32, num_workers=1)

    for i, (images, target) in tqdm(enumerate(train_loader), total=500, desc='Loader Progress'):
        if i == 500:
            break