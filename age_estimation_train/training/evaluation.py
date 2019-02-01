from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from .preprocessing import get_transformations
from .tensorboard import appa_real_log_figs
from .utils import get_label
from ..datasets.data_utils import *


class Evaluator:
    def __init__(self, dataset, writer, image_loss):
        self.dataset = dataset
        self.writer = writer
        self.logger = logging.getLogger(LOG_NAME)
        self.image_loss = image_loss

    def validate(self, model):
        model.eval()

        # Adult is class 1
        tp = 0  # Adults called adults
        tn = 0  # Children called children
        fp = 0  # Children called adults
        fn = 0  # Adults called children
        eps = 1e-8
        score = {}
        loss = 0
        loss_count = 0
        num = 0
        with torch.no_grad():
            for inputs, labels in self.dataset:
                inputs = inputs.to(DEVICE)
                gt = get_label(self.image_loss, labels)
                adults = labels['adult'].cpu().numpy().astype('bool')
                out = model.forward(inputs)
                debug = np.around(torch.exp(out[-1]).cpu().numpy()[:, 1], 3)
                debug = debug[debug < 0.05]
                num += len(debug)
                self.logger.info(f"{debug}")
                outputs = torch.exp(out[-1]).argmax(dim=1).to(torch.uint8).cpu().numpy().astype('bool')
                loss += self.image_loss(out, gt)
                tp += len(adults[(adults == 1) & (outputs == 1)])
                tn += len(adults[(adults == 0) & (outputs == 0)])
                fp += len(adults[(adults == 0) & (outputs == 1)])
                fn += len(adults[(adults == 1) & (outputs == 0)])
                loss_count += 1
            self.logger.info(f"Num of interest: {num}")
            loss /= max(loss_count, 1)
            acc = (tp + tn) / (tp + tn + fp + fn + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            self.logger.info(f"{'TP':4}|{'FP':4}")
            self.logger.info(f"{'FN':4}|{'TN':4}")
            self.logger.info(f"{tp:4}|{fp:4}")
            self.logger.info(f"{fn:4}|{tn:4}")
            self.logger.info(f"Acc \t{acc:.3f}")
            self.logger.info(f"Prec\t{precision:.3f}")
            self.logger.info(f"Rec \t{recall:.3f}")
            self.logger.info(f"F1  \t{f1:.3f}")
            score['acc'] = acc
            score['tp'] = tp
            score['fp'] = fp
            score['tn'] = tn
            score['fn'] = fn
            score['prec'] = precision
            score['rec'] = recall
            score['f1'] = f1
            score['loss'] = loss
        return score


def score_adience(model, writer, image_loss, subset='val', tag=''):
    logger = logging.getLogger(LOG_NAME)
    logger.info(f"Evaluating on Adience " + subset)
    trans = get_transformations()
    adience = AdienceDataset(transform=trans[1], subset=subset)
    data = DataLoader(adience, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)
    score(model, data, writer, image_loss=image_loss, to_log=0, tag=tag)


def score_appa_real(model, writer, image_loss, tag=''):
    logger = logging.getLogger(LOG_NAME)
    logger.info(f"APPA-REAL val")
    trans = get_transformations()
    appa = AppaRealDataset(transform=trans[1], subset='val/')
    data = DataLoader(appa, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)
    score(model, data, writer, image_loss=image_loss, to_log=0, tag=tag)


def score(model, data, writer, image_loss, to_log=-1, tag=''):
    """ Score a model on the given data. """
    logger = logging.getLogger(LOG_NAME)
    atg = get_age_to_group()

    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
        ])

    model.eval()
    mae = 0
    mae_count = 0

    off_by_idx = np.zeros(8, dtype='long')
    p_adults = 0
    n_adults = 0
    n_children = 0
    corr_minor = 0
    corr_adult = 0
    b_corr = 0

    loss = 0
    loss_count = 0
    with torch.no_grad():
        for inputs, labels in data:
            gt = get_label(image_loss, labels)
            t = get_label(NLLLoss(), labels).to(torch.float32)
            group = labels['group'].cpu().numpy().astype('long')
            adults = labels['adult'].cpu().numpy().astype('bool')
            n_adults += adults.sum()
            inputs = inputs.to(DEVICE)
            outputs = model.forward(inputs)
            loss += image_loss(outputs, gt)

            if isinstance(outputs, tuple):
                dist = outputs[0]
                pred = torch.matmul(torch.exp(dist), CLASSES).view(-1)
                b_preds = torch.exp(outputs[1]).argmax(dim=1).to(torch.uint8)
                b_corr += (b_preds.cpu().numpy() == adults).sum()
                p_group = torch.exp(outputs[2]).argmax(dim=1).to(torch.long)
                p_group = p_group.cpu().numpy()
                p_adults += torch.exp(outputs[1]).argmax(dim=1).to(torch.long).sum()

                corr_minor += ((p_group < 3) & (group < 3)).sum()
                corr_adult += ((p_group >= 3) & (group >= 3)).sum()
                delta = p_group - group
                for i in range(0, 8):
                    off_by_idx[i] += (np.abs(delta) <= i).sum()

                if loss_count == 0:
                    embed = torch.exp(outputs[0])
                    image = inputs.clone()
                    image = torch.cat([inv_normalize(i) for i in image]).view(-1, 3, 224, 224)
                    writer.add_embedding(embed, label_img=image, metadata=labels['group'].numpy().astype(str).tolist(),
                                         global_step=loss_count, tag='adience_group_' + tag + '_' + str(loss_count))

                    embed = torch.exp(outputs[2])
                    meta = p_group.astype(str).tolist()
                    writer.add_embedding(embed, label_img=image, metadata=meta,
                                         global_step=loss_count, tag='groups_' + tag)

                    embed = torch.exp(outputs[0])
                    meta = b_preds.cpu().numpy().astype(str).tolist()
                    writer.add_embedding(embed, label_img=image, metadata=meta,
                                         global_step=loss_count, tag='adult_' + tag)

                    embed = torch.exp(outputs[0])
                    meta = np.around(pred.cpu().numpy(), 2).astype(str).tolist()
                    writer.add_embedding(embed, label_img=image, metadata=meta,
                                         global_step=loss_count, tag='ages_' + tag)

            else:
                dist = outputs
                pred = torch.matmul(torch.exp(dist), CLASSES).view(-1)

                p_ages = pred.cpu().numpy().astype('long')
                p_group = np.zeros_like(p_ages)
                for i, j in enumerate(p_ages):
                    p_group[i] = atg[j]
                delta = p_group - group

                for i in range(0, 8):
                    off_by_idx[i] += (np.abs(delta) <= i).sum()

            if to_log > 0:
                if data.batch_size - to_log > 0:
                    idx = to_log
                else:
                    idx = data.batch_size
                l = {k: v for i, (k, v) in enumerate(labels.items()) if i < idx}
                appa_real_log_figs(dist[:idx], pred[:idx], l, writer, fold='offline')
                to_log -= idx

            mae += (t - pred).abs().sum()
            mae_count += len(pred)
            loss_count += 1
    loss /= max(loss_count, 1)
    for i in range(0, 8):
        logger.info(f"Off by {i}: {off_by_idx[i]}/{mae_count} = {(off_by_idx[i]/ mae_count):.4f}")
    n_children = mae_count - n_adults
    p_children = abs(mae_count - p_adults)

    logger.info(f"Adults: {p_adults}/{n_adults}")
    logger.info(f"Children: {p_children}/{n_children}")
    logger.info(f"Corr Adults: {corr_adult}/{n_adults}={(corr_adult/n_adults):.4f}")
    logger.info(f"Corr Child: {corr_minor}/{n_children}={(corr_minor / n_children):.4f}")
    logger.info(f"Acc minor: {corr_adult+corr_minor}/{n_adults+n_children}="
                f"{((corr_adult+corr_minor)/(n_adults+n_children)):.4f}")
    logger.info(f"Binary Acc: {(b_corr/mae_count):.4f}")
    score = (mae/max(mae_count, 1)).type(torch.FloatTensor).item()
    writer.add_scalar(f'mae', score)
    writer.add_scalar(f'loss', loss)

    return score, loss


def score_dir(model, data, writer, image_loss, to_log=-1, tag=''):
    """ Score a model on the given data. """
    logger = logging.getLogger(LOG_NAME)
    atg = get_age_to_group()

    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
        ])

    model.eval()
    p_adults = 0
    n_adults = 0
    n_children = 0
    corr_minor = 0
    corr_adult = 0
    b_corr = 0

    count = 0
    mae_count = 0
    with torch.no_grad():
        for inputs, labels in data:
            group = labels['group'].cpu().numpy().astype('long')
            adults = labels['adult'].cpu().numpy().astype('bool')
            n_adults += adults.sum()

            inputs = inputs.to(DEVICE)
            outputs = model.forward(inputs)

            if isinstance(outputs, tuple):
                dist = outputs[0]
                pred = torch.matmul(torch.exp(dist), CLASSES).view(-1)

                b_preds = torch.exp(outputs[1]).argmax(dim=1).to(torch.uint8)
                b_corr += (b_preds.cpu().numpy() == adults).sum()

                p_group = torch.exp(outputs[2]).argmax(dim=1).to(torch.long)
                p_group = p_group.cpu().numpy()

                p_adults += torch.exp(outputs[1]).argmax(dim=1).to(torch.long).sum()

                corr_minor += ((p_group < 3) & (group < 3)).sum()
                corr_adult += ((p_group >= 3) & (group >= 3)).sum()

                if count == 0:
                    image = inputs.clone()
                    image = torch.cat([inv_normalize(i) for i in image]).view(-1, 3, 224, 224)

                    embed = torch.exp(outputs[2])
                    meta = p_group.astype(str).tolist()
                    writer.add_embedding(embed, label_img=image, metadata=meta,
                                         global_step=count, tag='groups_' + tag)

                    embed = torch.exp(outputs[0])
                    meta = b_preds.cpu().numpy().astype(str).tolist()
                    writer.add_embedding(embed, label_img=image, metadata=meta,
                                         global_step=count, tag='adult_' + tag)

                    embed = torch.exp(outputs[0])
                    meta = np.around(pred.cpu().numpy(), 2).astype(str).tolist()
                    writer.add_embedding(embed, label_img=image, metadata=meta,
                                         global_step=count, tag='ages_' + tag)
                count += 1
            else:
                dist = outputs
                pred = torch.matmul(torch.exp(dist), CLASSES).view(-1)

            if to_log > 0:
                if data.batch_size - to_log > 0:
                    idx = to_log
                else:
                    idx = data.batch_size
                l = {k: v for i, (k, v) in enumerate(labels.items()) if i < idx}
                appa_real_log_figs(dist[:idx], pred[:idx], l, writer, fold='offline')
                to_log -= idx
            mae_count += len(pred)

    n_children = mae_count - n_adults
    p_children = abs(mae_count - p_adults)
    n_adults = max(n_adults, 1e-5)
    n_children = max(n_children, 1e-5)

    logger.info(f"Adults: {p_adults}/{n_adults}")
    logger.info(f"Children: {p_children}/{n_children}")
    logger.info(f"Corr Adults: {corr_adult}/{n_adults}={(corr_adult/n_adults):.4f}")
    logger.info(f"Corr Child: {corr_minor}/{n_children}={(corr_minor / n_children):.4f}")
    logger.info(f"Acc minor: {corr_adult+corr_minor}/{n_adults+n_children}="
                f"{((corr_adult+corr_minor)/(n_adults+n_children)):.4f}")
    logger.info(f"Binary Acc: {(b_corr/mae_count):.4f}")

    return score


def pred_dir(model, data, keep=0):
    model.eval()
    buffer = []
    buffer.append(f"path, out")
    with torch.no_grad():
        for inputs, labels in data:
            inputs = inputs.to(DEVICE)
            outputs = model.forward(inputs)
            outputs = outputs[0].cpu().numpy(), outputs[1].cpu().numpy(), outputs[2].cpu().numpy()
            # for path, dist, padult, pgroup in zip(labels['path'], outputs[0], outputs[1], outputs[2]):
            #   buffer.append(f"{path}, {torch.exp(dist)}, {torch.exp(padult)}, {torch.exp(pgroup)}")
            for path, out in zip(labels['path'], outputs[keep]):
                out = np.exp(out)
                txt = " ".join(np.around(out, 3).astype(str).tolist())
                buffer.append(f"{path}, {txt}")

    return buffer
