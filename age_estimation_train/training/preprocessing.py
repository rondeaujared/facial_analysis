from torchvision import transforms


class ImdbTransformer(object):

    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, img, face=None):
        # deltas = np.round((np.random.rand(4) - 0.5) * 20).astype(int)
        # m = max(img.size[0]*MARGIN, img.size[1]*MARGIN)
        margin = 0.3  # np.random.rand()*0.4+0.1
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

