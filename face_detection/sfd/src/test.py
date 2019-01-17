from __future__ import print_function

import glob

import cv2
import os
from scipy import misc

from torch.utils.data.dataloader import DataLoader
from face_detection.sfd.src.detect import *
from face_detection.sfd.models import net_s3fd
from .utils import LOGGER, image_resize, EvalDirectory, DEVICE

weights = '/home/research/jrondeau/research/risp-prototype/engine/face_detection/sfd/data/s3fd_convert.pth'
toglob = set(['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'])
MIN_SCORE = 0.90


def find_faces_fast(root, display=False):
    logger = LOGGER
    net = getattr(net_s3fd, 's3fd')()
    net.load_state_dict(torch.load(weights))
    net = torch.nn.DataParallel(net)
    net = net.to(DEVICE)
    net.eval()
    dataset = EvalDirectory(root)
    dataset = DataLoader(dataset, batch_size=64, pin_memory=True, shuffle=False, num_workers=0)
    faces = []
    output = os.getcwd() + '/data/'

    with torch.no_grad():
        for image, paths in dataset:
            olist = net(image)
            bboxlist = detect_fast(olist)
            for ix in range(bboxlist.shape[1]):
                curr = bboxlist[:, ix, :]
                path = paths[ix]
                keep = nms(curr, 0.30)
                curr = curr[keep, :]
                for b in curr:
                    x1, y1, x2, y2, s = b
                    if s < MIN_SCORE:
                        continue
                    else:
                        _str = f"{path} {int(x1)} {int(y1)} {int(x2)} {int(y2)} {s:.2f}"
                        logger.info(_str)
                        faces.append((path, int(x1), int(y1), int(x2), int(y2)))

                if display:
                    print(f"Path: {path} typeof: {type(path)}")
                    img = cv2.imread(path)
                    imgshow = image_resize(img)
                    for b in curr:
                        x1, y1, x2, y2, s = b

                        if s < MIN_SCORE:
                            continue
                        else:
                            cv2.rectangle(imgshow, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

                    toshow = cv2.cvtColor(imgshow, cv2.COLOR_BGR2RGB)
                    try:
                        misc.imshow(toshow)
                    except:
                        pass
                    out_path = path.replace(root, '')
                    out_dir = output + out_path[:out_path.rfind('/')]
                    print(f"out dir: {out_dir}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = output + out_path + '_output.png'
                    print(f"Saving to {out_path}")
                    cv2.imwrite(out_path, imgshow)

        return faces


def find_faces(dir, display=False, device_ids=None):
    """
    @TODO: Figure out why this warning is being thrown (does not appear to be changing results)
    "Unescaped left brace in regex is deprecated, passed through in regex; marked by <-- HERE in m/%{ <-- HERE (.*?)}/ at /usr/bin/see line 528.
    Error: no "view" rule for type "image/png" passed its test case"
    :param dir:
    :param display:
    :param device_ids:
    :return:
    """
    logger = LOGGER
    net = getattr(net_s3fd, 's3fd')()
    net.load_state_dict(torch.load(weights))
    logger.info(f"Device ID: {device_ids}")
    net = net.cuda(device_ids).eval()
    net = torch.nn.DataParallel(net, device_ids=(device_ids,))  # If training on subset of GPUs use: device_ids=[0, 1, 2, 3])
    todo = [dir]
    output = os.getcwd() + '/data/'
    faces = []

    with torch.no_grad():
        while todo:
            curr = todo.pop()
            logger.info(f"Parsing dir {curr}")
            for c in os.listdir(curr):
                t = os.path.join(curr, c)
                if os.path.isdir(t):
                    todo.append(t)
            hits = []
            for ext in toglob:
                a = os.path.join(curr, '*' + ext)
                hits.extend(glob.glob(a))

            for file in hits:
                img = cv2.imread(file)
                img = image_resize(img)
                try:
                    bboxlist = detect(net, img)
                except Exception as e:
                    logger.warning(f"Failed to detect image {file}; with error {e}.")
                    continue
                keep = nms(bboxlist, 0.30)
                bboxlist = bboxlist[keep, :]
                for b in bboxlist:
                    x1, y1, x2, y2, s = b
                    if s < MIN_SCORE:
                        continue
                    else:
                        p = os.path.join(curr, file)
                        logger.info(f"{p} {int(x1)} {int(y1)} {int(x2)} {int(y2)} {s:.2f}")
                        faces.append((p, int(x1), int(y1), int(x2), int(y2)))

                if display:
                    logger.info(f"On image {file}")
                    imgshow = np.copy(img)
                    for b in bboxlist:
                        x1, y1, x2, y2, s = b

                        if s < MIN_SCORE:
                            logger.info(f"Excluding face with score {s}")
                            continue
                        else:
                            cv2.rectangle(imgshow, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

                    toshow = cv2.cvtColor(imgshow, cv2.COLOR_BGR2RGB)
                    try:
                        misc.imshow(toshow)
                    except:
                        pass
                    out_path = file.replace(dir, '')
                    cv2.imwrite(output + out_path +'_output.png', imgshow)

    return faces
