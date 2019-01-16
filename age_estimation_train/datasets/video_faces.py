import torch.utils.data as data
from torch.utils.data import Sampler, RandomSampler, BatchSampler
from torchvision.datasets.folder import default_loader

from settings import *

# BASE_DIR = '/mnt/data/playground/umdfaces/video_frames/cropped/'
BASE_DIR = '/mnt/data/playground/YouTubeFaces/YouTubeFaces/aligned_images_DB/'


def get_video_faces_paths(dir=BASE_DIR, subsample=None):
    extensions = ['.jpg', '.png']
    dir = os.path.expanduser(dir)
    people = {}

    for target in sorted(os.listdir(dir))[:subsample]:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        people[target] = {}

        for root, _, fnames in sorted(os.walk(d)):
            vid = root[root.rfind('/')+1:]

            for fname in sorted(fnames):
                fn = fname.lower()
                if any(fn.endswith(ext) for ext in extensions):
                    people[target][vid] = people[target].get(vid, []) + [fname]

    return people


class VideoFacesSampler(BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, vids, batch_size=64, frames_per_vid=8):
        super().__init__(sampler, batch_size=batch_size, drop_last=False)

        # vids[i] -> ( (person, video), [frame_1, frame_2, ... ])
        self.vids = vids
        self.frames_per_vid = frames_per_vid

    def __iter__(self):
        # batch[i] -> ( (person, video), [frame_1, frame_2, ... ])
        batch = []
        vids_per_batch = self.batch_size // self.frames_per_vid
        sample_iter = iter(self.sampler)

        empty = False
        while not empty:

            while len(batch) < self.batch_size and not empty:
                idx = next(sample_iter, None)
                if idx is None:
                    empty = True
                    break

                info, frames = self.vids[idx]
                n_frames = len(frames)

                if n_frames >= self.frames_per_vid:
                    sample = random.sample(frames, self.frames_per_vid)
                    for frame in sample:
                        batch.append((frame, idx))
                else:
                    pass

            assert len(batch) <= self.batch_size, f"Batch size too large?!"
            yield batch
            batch = []


class VideoFacesDataset(data.Dataset):

    def __init__(self, root=BASE_DIR, transform=None, target_transform=None,
                 subsample=None, batch_size=64, sampler=RandomSampler, frames_per_vid=8):

        assert transform is not None, f"transform is None {transform}"

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.logger = logging.getLogger(LOG_NAME)

        # vids[i] -> ( (person, video), [frame_1, frame_2, ... ])
        vids = {}
        images = []
        data = get_video_faces_paths(BASE_DIR, subsample)

        i = 0
        vc = 0
        for k, val in data.items():
            person = k

            for l, v in val.items():
                paths = list(map(lambda x: BASE_DIR + person + '/' + str(l) + '/' + str(x), v))
                vids[vc] = ((person, l), paths)
                vc += 1
                for img in paths:
                    images.append(img)
            i += 1

        self.vids = vids
        self.images = images

        self.logger.info(f"{len(self.vids)} videos, {len(self.images)} frames, {i} identities in aux dataset")

        self.sampler = VideoFacesSampler(sampler(self), self.vids, batch_size=batch_size, frames_per_vid=frames_per_vid)

    def __getitem__(self, index):
        """
        Args:
            index tuple(image_path, person_key): Index
        Returns:
            tuple: (image, file_path, vid_id)
        """
        file, vid = index
        frame = self.loader(file)

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, file, vid

    def __len__(self):
        return len(self.vids)
