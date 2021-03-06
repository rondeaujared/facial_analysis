import glob
import pickle
import os
import torch

from face_detection.sfd.src.utils import EvalDirectory
from age_estimation_train.models.AuxilliaryAgeNet import AuxAgeNet
from age_estimation_train.training.evaluation import pred_dir
from age_estimation_train.training.preprocessing import get_transformations
from age_estimation_train.training.utils import setup_custom_logger
from face_detection.sfd.src.test import find_faces_fast
from icrawler.builtin import (FlickrImageCrawler, GoogleImageCrawler)

root = '/mnt/fastdata/datasets/age-crawler/'
logger = setup_custom_logger('child_crawler', 'child_crawler')


def google_crawler(terms, filter, offset=0):
    for term in terms:
        google_crawler = GoogleImageCrawler(feeder_threads=2, parser_threads=2, downloader_threads=24,
                                            storage={'root_dir': root + 'google/' + term.replace(' ', '_')})

        google_crawler.crawl(keyword=term, max_num=1000, file_idx_offset=offset, filters=filter)


def flickr_crawler(terms):
    for tag in terms:
        crawler = FlickrImageCrawler(apikey='974e364d51e2f09a3bb24c9c92f3f0f1',
                                     feeder_threads=2, parser_threads=2, downloader_threads=24,
                                     storage={'root_dir': root + 'flickr/' + tag.replace(' ', '_')})
        crawler.crawl(max_num=1000, tags=tag, tag_mode='all')


def get_faces(path):
    faces = find_faces_fast(path, logger)
    pfaces = []
    for face in faces:
        pfaces.append(face[0])
    return faces, pfaces


def get_all(path):
    toglob = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    hits = glob.glob(path + '**', recursive=True)
    all = [x for x in hits if x[x.rfind('.'):] in toglob]
    return all


def crawl(terms):
    years = list(range(2000, 2020))[::-1]
    first = True
    for year in years:
        filter = dict(
            date=((year, 1, 1), (year, 12, 30))
        )
        if first:
            offset = 0
            first = False
        else:
            offset = 'auto'
        google_crawler(terms, filter, offset)
    flickr_crawler(terms)


def detect_faces(dir):
    faces_pfaces = get_faces(dir)
    pickle.dump(faces_pfaces, open(dir + "faces.pickle", "wb"))
    all = get_all(dir)
    pickle.dump(all, open(dir + "all.pickle", "wb"))


def prune_faceless(dir):
    faces_pfaces = pickle.load(open(dir + "faces.pickle", "rb"))
    all = pickle.load(open(dir + "all.pickle", "rb"))
    set_pfaces = set(faces_pfaces[1])
    set_all = set(all)
    set_keep = set_pfaces.intersection(set_all)
    set_remove = set_all.difference(set_pfaces)

    for f in set_remove:
        new_f = f.replace('age-crawler', 'age-crawler/faceless')
        new_f = new_f[:new_f.rfind('/')]
        os.makedirs(new_f, exist_ok=True)
        os.rename(f, new_f + f[f.rfind('/'):])


def get_ages(dir, fname):
    from torch.utils.data import DataLoader
    _WEIGHTS = { 'hmm': '/home/research/jrondeau/research/risp-prototype/engine/age_estimation/age_estimation_train/archive/Jan08_IMDB_150_EPOCHS/debugging.pthrecent'}
    _MODEL_PARAMS = {'base_weights': '', 'imagenet': True, 'freeze_features': True,
                     'drop_rate': 0.20, 'aux_weights': _WEIGHTS['hmm']}
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = pickle.load(open(dir + "faces.pickle", "rb"))
    trans = get_transformations()
    data = EvalDirectory(dir, trans[1], cache=d[0])
    data = DataLoader(data, batch_size=512, pin_memory=True, shuffle=False, num_workers=24)
    model = AuxAgeNet(**_MODEL_PARAMS)
    model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    buffer = pred_dir(model, data, keep=1)
    f = open(f"{fname}.txt", "w")
    f.write("\n".join(buffer))


if __name__ == '__main__':
    """
    The logic here is as following:
    1) Define search terms
    2) Crawl websites and save images according to structure root/google/term1/img1.jpg etc.
    3) Perform face detection
    4) Remove images with no detected faces
    5) Remove duplicates
    6) Prune images with no children
    :return:
    """
    terms = ['child', 'infant', 'toddler', 'teen', 'teenager', 'baby',
             '0 year old', '1 year old', '2 year old', '3 year old', '4 year old', '5 year old',
             '6 year old', '7 year old', '8 year old', '9 year old', '10 year old', '11 year old', '12 year old']
    # crawl(terms)
    base = '/mnt/fastdata/datasets/age-crawler/google/'
    todo = list(map(lambda x: base + x.replace(' ', '_') + '/', terms))

    for term in todo:
        detect_faces(dir=term)

    for term in terms:
        dir = base + term.replace(' ', '_') + '/'
        prune_faceless(dir)

    for term in terms:
        get_ages(base + term.replace(' ', '_') + '/', term.replace(' ', '_'))
