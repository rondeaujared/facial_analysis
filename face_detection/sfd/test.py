from __future__ import print_function

import glob

import cv2
import os
from scipy import misc

from .detect import *
from .models import net_s3fd

weights = '/home/research/jrondeau/research/risp-prototype/engine/face_detection/sfd/data/s3fd_convert.pth'
images = {'.jpg': True, '.png': True, '.jpeg': True}


def find_faces(dir, logger, display=False, device_ids=None):
    net = getattr(net_s3fd, 's3fd')()
    net.load_state_dict(torch.load(weights))
    print(f"Device ID: {device_ids}")
    net = net.cuda(device_ids).eval()
    net = torch.nn.DataParallel(net, device_ids=(device_ids,))  # If training on subset of GPUs use: device_ids=[0, 1, 2, 3])
    todo = [dir]
    output = os.getcwd() + '/data/'
    faces = []
    with torch.no_grad():
        while todo:
            curr = todo.pop()
            # os.chdir(curr)
            for c in os.listdir(curr):
                t = os.path.join(curr, c)
                if os.path.isdir(t):
                    todo.append(t)

            for file in glob.glob(curr + "/*.jpg") + glob.glob(curr + "/*.png") + glob.glob(curr + "/*.jpeg"):
                keep = images.get(file[file.find('.'):], False)
                if not keep:
                    continue
                img = cv2.imread(file)

                try:
                    bboxlist = detect(net, img)
                except Exception as e:
                    logger.warning(f"Failed to detect image {file}; with error {e}.")
                    continue
                keep = nms(bboxlist, 0.30)
                bboxlist = bboxlist[keep, :]
                for b in bboxlist:
                    x1, y1, x2, y2, s = b
                    if s < 0.90:
                        continue
                    else:
                        p = os.path.join(curr, file)
                        logger.info(f"{p} {int(x1)} {int(y1)} {int(x2)} {int(y2)} {s:.2f}")
                        faces.append((p, int(x1), int(y1), int(x2), int(y2)))

                if display:
                    imgshow = np.copy(img)
                    for b in bboxlist:
                        x1, y1, x2, y2, s = b

                        if s < 0.5:
                            continue
                        else:
                            cv2.rectangle(imgshow, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

                    toshow = cv2.cvtColor(imgshow, cv2.COLOR_BGR2RGB)
                    try:
                        misc.imshow(toshow)
                    except:
                        pass
                    cv2.imwrite(output + file[:-4] +'_output.png', imgshow)

    return faces
