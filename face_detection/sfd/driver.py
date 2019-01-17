import torch
from face_detection.sfd.src.test import find_faces, find_faces_fast
torch.backends.cudnn.bencmark = True


if __name__ == "__main__":
    # dir = '/mnt/data/playground/youtube-minors/thumbnails'
    # dir = '/mnt/data/playground/challenging-binary-age'
    # dir = '/mnt/fastdata/datasets/age-crawler/labelled_flickr/'
    # dir = '/mnt/data/playground/redlight/images/train/nsfw/'
    dir = '/mnt/fastdata/datasets/age-crawler/labelled_flickr/baby/adult/'
    find_faces_fast(dir, display=False)
