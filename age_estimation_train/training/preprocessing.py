from torchvision import transforms
import torch
import cv2
import numpy as np


class ImdbTransformer(object):

    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, img, face=None, crop=True):
        # deltas = np.round((np.random.rand(4) - 0.5) * 20).astype(int)
        # m = max(img.size[0]*MARGIN, img.size[1]*MARGIN)

        if crop:
            margin = np.random.rand()*0.4+0.1
            m = max((face['x2']-face['x1']) * margin, (face['y2'] - face['y1']) * margin)
            img = img.crop((face['x1']-m,
                            face['y1']-m,
                            face['x2']+m,
                            face['y2']+m))
        img = self.transforms(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get_transformations():
    mean = [0.485, 0.456, 0.406]
    stdd = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([

        #transforms.ColorJitter(brightness=0.5, contrast=0.5,
        #                       saturation=0.5, hue=0),
        #transforms.RandomAffine(degrees=180, translate=(0.1, 0.1)),
        #transforms.RandomGrayscale(p=0.2),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.50),
        #transforms.RandomVerticalFlip(p=0.50),
        transforms.ToTensor(),
        transforms.Normalize(mean, stdd)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(224),#, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, stdd)
    ])

    return train_transforms, test_transforms


class S3fdTransformer(object):
    def __init__(self, static):
        self.static = static
        self.S3FD = True
        from settings import S3FD_RESIZE
        self.big_side = S3FD_RESIZE

    def __call__(self, path):
        img = cv2.imread(path)
        if img is None:
            self.logger.warning(f"Image opened as none! {path}")
        img = self.image_resize(img, big_side=self.big_side, static=self.static)

        if img is None or img.shape[0] > 4000 or img.shape[1] > 4000:
            self.logger.warning(f"Image too large: {img.shape}; ignoring")
            return np.zeros((1, 5))

        try:
            img = img - np.array([104, 117, 123])
            p = np.random.rand()
            if p > 0.50 and not self.static:
                img = cv2.flip(img, flipCode=1)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).to(dtype=torch.float)
        except Exception as e:
            self.logger.warning(f"Failed to detect image {path}; with error {e}.")
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    @staticmethod
    def image_resize(image, width=None, height=None, big_side=640, inter=cv2.INTER_AREA, static=False):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            if max(h, w) < big_side:
                hpad = big_side - h
                wpad = big_side - w
                p = np.random.rand()
                if p > 0.50 and not static:
                    image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad,
                                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    image = cv2.copyMakeBorder(image, left=wpad, top=hpad, bottom=0, right=0,
                                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
                return image
            if h > w:
                height = big_side
            else:
                width = big_side

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        image = cv2.resize(image, dim, interpolation=inter)
        (h, w) = image.shape[:2]
        if width is None:  # Height was big side
            wpad = big_side - w
            hpad = 0
        else:
            wpad = 0
            hpad = big_side - h

        p = np.random.rand()
        if p > 0.50 and not static:
            image = cv2.copyMakeBorder(image, left=0, top=0, bottom=hpad, right=wpad,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            image = cv2.copyMakeBorder(image, left=wpad, top=hpad, bottom=0, right=0,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image
