import torch.utils.data as data
from torchvision.datasets.folder import default_loader

from face_detection.sfd.test import find_faces
from settings import *
from ..training.preprocessing import ImdbTransformer


class EvalDirectory(data.Dataset):

    def __init__(self, root, transform, cache=None):
        self.root = root
        self.transform = ImdbTransformer(transform)
        self.loader = default_loader
        self.logger = logging.getLogger(LOG_NAME)
        self.labels = []
        if cache is None:
            print(f"Empty cache; finding faces")
            self.data = find_faces(self.root, self.logger, display=False)
        else:
            self.data = cache

        for img in self.data:
            path = img[0]
            label = {}
            label['path'] = path
            if 'child' in path:
                label['group'] = 0
                label['adult'] = 0
            elif 'adult' in path:
                label['group'] = 4
                label['adult'] = 1
            elif 'nsfw' in path:
                label['group'] = 4
                label['adult'] = 1
            else:
                label['group'] = -1
                label['adult'] = -1
            self.labels.append(label)

    def __getitem__(self, index):
        path, x1, y1, x2, y2 = self.data[index]
        face = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img, face)

        return img, self.labels[index]

    def __len__(self):
        return len(self.data)
