import logging
import sys

from io import StringIO

from .consistency_losses import *


def get_label(image_loss, labels):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(image_loss, KLDivLoss):
        gt = labels['label'].to(DEVICE, dtype=torch.float32)
    elif isinstance(image_loss, NLLLoss):
        gt = labels['app_age'].to(DEVICE, dtype=torch.long)
    elif image_loss.__name__ == mean_ldl_loss.__name__:  # mean_ldl_loss
        gt = labels['label'].to(DEVICE, dtype=torch.float32), labels['app_age'].to(DEVICE, dtype=torch.float32)
    elif image_loss.__name__ == child_adience_loss.__name__:
        gt = labels['adult'].to(DEVICE, dtype=torch.long), labels['group'].to(DEVICE, dtype=torch.long)
    elif image_loss.__name__ == child_adience_ldl_loss.__name__:
        gt = labels['label'].to(DEVICE, dtype=torch.float32), \
             labels['adult'].to(DEVICE, dtype=torch.long), \
             labels['group'].to(DEVICE, dtype=torch.long)
    else:
        raise Exception(f"Invalid criterion: {image_loss.__name__}")
    return gt


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def catch_stdin(func, *args):
    capture = StringIO()
    save_stdout = sys.stdout
    sys.stdout = capture
    func(*args)
    sys.stdout = save_stdout
    msg = capture.getvalue()

    return msg


def setup_custom_logger(name, file_name, level='DEBUG'):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(file_name, mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    screen_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)

    logger.debug(f"Saving logger {name} to {file_name}.log")
    return logger
