import torch.nn.functional as F
from face_detection.sfd.src.bbox import *
from .utils import LOGGER

logger = LOGGER
FILTER = 0.50


def detect_fast(olist):
    bboxlist = []
    for i in range(len(olist)//2):
        olist[i*2] = F.softmax(olist[i*2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]

    for i in range(len(olist)//2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        # FB, FC, FH, FW = ocls.size()   # feature map size
        stride = 2**(i+2)           # 4,8,16,32,64,128
        poss = zip(*np.where(ocls[:, 1, :, :] > FILTER))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride/2+windex*stride, stride/2+hindex*stride
            score = ocls[:, 1, hindex, windex]
            loc = None
            loc = oreg[:, :, hindex, windex].contiguous().view(-1, 4)

            priors = torch.Tensor([[axc/1.0, ayc/1.0, stride*4/1.0, stride*4/1.0]])
            variances = [0.1, 0.2]
            if loc is None:
                continue
            box = decode(loc, priors, variances)

            bboxlist.append(torch.cat((box, score.view(-1, 1)), 1).numpy())
        #box = decode(loc, priors, variances)

        #bboxlist.append(torch.cat((box, score.view(-1, 1)), 1).numpy())

    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        print(f"No faces deteted; not sure if this will work")
        bboxlist = None  # np.zeros((1, 1, 5))

    """ @TODO: turn this method into gpu
        stride = torch.Tensor([2 ** (i + 2)]).to(torch.device('cuda'), dtype=torch.float)
        poss = zip(*torch.nonzero(ocls[:, 1, :, :] > FILTER))
        a = torch.nonzero(ocls[:, 1, :, :] > FILTER)
        score = ocls[:, 1, a[:, 1], a[:, 2]]
        loc = oreg[:, :, a[:, 1], a[:, 2]].contiguous().view(-1, 4)
        a = a.to(torch.float)
        axc, ayc = stride / 2 + a[:, 2] * stride, stride / 2 + a[:, 1] * stride
        b = torch.stack([axc, ayc], dim=1)
        c = torch.stack([b, stride*4], dim=2)
        d = torch.stack([c, stride * 4], dim=2)
        priors = torch.Tensor([axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0])
        variances = [0.1, 0.2]
    """
    return bboxlist


def detect(net, img):
    if img is None or img.shape[0] > 4000 or img.shape[1] > 4000:
        logger.warning(f"Image too large: {img.shape}; ignoring")
        return np.zeros((1, 5))

    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = torch.from_numpy(img).to(dtype=torch.float)
    # img = img.to(dtype=torch.float, device=device)

    olist = net(img)

    bboxlist = []
    for i in range(len(olist)//2):
        olist[i*2] = F.softmax(olist[i*2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]

    for i in range(len(olist)//2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()   # feature map size
        stride = 2**(i+2)           # 4,8,16,32,64,128
        anchor = stride*4
        poss = zip(*np.where(ocls[:, 1, :, :] > FILTER))

        for Iindex, hindex, windex in poss:
            axc, ayc = stride/2+windex*stride, stride/2+hindex*stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)

            priors = torch.Tensor([[axc/1.0, ayc/1.0, stride*4/1.0, stride*4/1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1, y1, x2, y2, score])

    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))

    return bboxlist
