import torch
from face_detection.sfd.src.test import find_faces, find_faces_fast
torch.backends.cudnn.bencmark = True


if __name__ == "__main__":
    # dir = '/mnt/data/playground/youtube-minors/thumbnails'
    # dir = '/mnt/data/playground/challenging-binary-age'
    # dir = '/mnt/fastdata/datasets/age-crawler/labelled_flickr/'
    # dir = '/mnt/data/playground/redlight/images/train/nsfw/'
    #dir = '/mnt/fastdata/datasets/age-crawler/labelled_flickr/baby/adult/'
    #dir = '/mnt/nfs/scratch1/jrondeau/datasets/challenging-binary-age/'
    dir = '/home/jared/research/lab-backup/challenging-binary-age/'
    find_faces_fast(dir, batch_size=4, display=True)
