import glob
import pickle

import os
import torch
from engine.age_estimation.age_estimation_train.datasets.eval_directory import EvalDirectory
from engine.age_estimation.age_estimation_train.models.AuxilliaryAgeNet import AuxAgeNet
from engine.age_estimation.age_estimation_train.training.evaluation import pred_dir
from engine.age_estimation.age_estimation_train.training.preprocessing import get_transformations
from engine.age_estimation.age_estimation_train.training.utils import setup_custom_logger
from engine.face_detection.sfd.test import find_faces
from icrawler.builtin import (FlickrImageCrawler, GoogleImageCrawler)

root = '/mnt/fastdata/datasets/age-crawler/'
logger = setup_custom_logger('child_crawler', 'child_crawler')

"""
@TODO: idk if this will be neccesary to multi-thread

def process(path, batch_size=128):
    logger.debug("Processing path: %s, valid: %s" % (path, os.path.isdir(path)))
    # Shared resources for our threads, do not reassign these
    todo = []

    # Synchronization Objects
    out_of_frames = Event()
    faces_ready = Event()

    # Don't forget to change based on len(threads) + 1
    shared_frames_ready = Barrier(4, timeout=120)
    finished_processing = Barrier(4, timeout=120)

    # Initialize Threads
    threads = [
        Thread(
            target=ages_thread, name="AG_Processing",
            kwargs=dict(
                shared_frames=shared_frames, faces_ready=faces_ready,
                frames_ready=shared_frames_ready, finished_barrier=finished_processing,
                is_done=out_of_frames
            )
        )
    ]

    [thread.start() for thread in threads]
    logger.debug("Started threads")

    for frame_batch in load(path, batch_size=batch_size):
        shared_frames.clear()
        shared_frames.extend(frame_batch)
        shared_frames_ready.wait()
        logger.debug('Shared frames extended.')

        finished_processing.wait()
        logger.debug('Cycle complete.')
    else:
        out_of_frames.set()
        shared_frames.clear()
        shared_frames_ready.wait()
        logger.debug('Load Complete.')

    [thread.join() for thread in threads]
    logger.debug("Joined threads")
"""


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


def get_faces(path, gpu=0):
    faces = find_faces(path, logger, device_ids=gpu)
    pfaces = []
    for face in faces:
        pfaces.append(face[0])
    return faces, pfaces


def get_all(path):
    images = {'.jpg': True, '.png': True, '.jpeg': True}
    todo = [path]
    all = []
    while todo:
        curr = todo.pop()
        os.chdir(curr)
        for c in os.listdir(curr):
            t = os.path.join(curr, c)
            if os.path.isdir(t):
                todo.append(t)

        for file in glob.glob("*.jpg") + glob.glob("*.png"):
            keep = images.get(file[file.find('.'):], False)
            if not keep:
                continue
            p = os.path.join(curr, file)
            all.append(p)
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


def detect_faces(dir, gpu):
    faces_pfaces = get_faces(dir, gpu)
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
    model = torch.nn.DataParallel(model)  # If training on subset of GPUs use: device_ids=[0, 1, 2, 3])
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

    import threading
    start_nthreads = len(threading.enumerate())
    while todo:
        threads = []
        for i in range(4):
            if todo:
                t = todo.pop()
                threads.append(threading.Thread(target=detect_faces, name="face_detect", kwargs=dict(dir=t, gpu=i)))

        [thread.start() for thread in threads]
        print(threading.enumerate())
        nthreads = len(threading.enumerate()) - start_nthreads
        print(f"nthreads: {nthreads}")
        [thread.join() for thread in threads]

    print(f"Past Pool")
    for term in terms:
        dir = base + term.replace(' ', '_') + '/'
        prune_faceless(dir)

    for term in terms:
        get_ages(base + term.replace(' ', '_') + '/', term.replace(' ', '_'))
