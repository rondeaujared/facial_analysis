from datetime import datetime
from tensorboardX import SummaryWriter
from .datasets.data_utils import *
from .datasets.eval_directory import EvalDirectory
from .models.AuxilliaryAgeNet import AuxAgeNet
from .training.evaluation import *
from .training.utils import setup_custom_logger
import settings


def eval():
    # Setup logging
    experiment_name = 'test'
    start_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tensorboard_dir = os.path.join('runs', start_time + '_' + experiment_name)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger = setup_custom_logger(LOG_NAME, tensorboard_dir + '/' + experiment_name + '.log', level='INFO')
    save_path = tensorboard_dir + '/' + experiment_name + '.pth'

    # Setup model
    trans = get_transformations()
    model = AuxAgeNet(**settings.MODEL_PARAMS)
    model = torch.nn.DataParallel(model)  # If training on subset of GPUs use: device_ids=[0, 1, 2, 3])
    model = model.to(DEVICE)

    '''
    ###########
    # Adience #
    ###########
    logger.info(f"Adience val")
    adience = AdienceDataset(transform=trans[1], subset='val')
    data = DataLoader(adience, batch_size=256, pin_memory=True, shuffle=False, num_workers=24)
    score(model, data, writer, image_loss=child_adience_ldl_loss, to_log=0, tag='adience_val')

    logger.info(f"Adience test")
    adience = AdienceDataset(transform=trans[1], subset='test')
    data = DataLoader(adience, batch_size=256, pin_memory=True, shuffle=False, num_workers=24)
    score(model, data, writer, image_loss=child_adience_ldl_loss, to_log=0, tag='adience_test')
    

    #############
    # APPA-REAL #
    #############
    logger.info(f"APPA-REAL val")
    appareal = AppaRealDataset(transform=trans[1], subset='val/', ext='', crop_faces=True)
    data = DataLoader(appareal, batch_size=256, pin_memory=True, shuffle=False, num_workers=24)
    score(model, data, writer, image_loss=child_adience_ldl_loss, to_log=0, tag='appa_val')

    logger.info(f"APPA-REAL test")
    appareal = AppaRealDataset(transform=trans[1], subset='test/', ext='', crop_faces=True)
    data = DataLoader(appareal, batch_size=256, pin_memory=True, shuffle=False, num_workers=24)
    score(model, data, writer, image_loss=child_adience_ldl_loss, to_log=0, tag='appa_test')
    '''
    #############
    # Directory #
    #############
    start = os.getcwd()
    #dirs = ['/mnt/data/playground/challenging-binary-age/child',
    #        '/mnt/data/playground/redlight/images/test/nsfw']
    dirs = ['/mnt/fastdata/datasets/age-crawler/labelled_flickr']
    for dir in dirs:
        logger.info(f"Parsing directory {dir}")
        data = EvalDirectory(dir, trans[1])
        data = DataLoader(data, batch_size=512, pin_memory=True, shuffle=False, num_workers=24)
        os.chdir(start)
        # score_dir(model, data, writer, image_loss=child_adience_ldl_loss, to_log=0, tag='nsfw')
        buf1 = pred_dir(model, data, keep=0)
        buf2 = pred_dir(model, data, keep=1)
        buf3 = pred_dir(model, data, keep=2)
        logger.info(buf1)
        logger.info(buf2)
        logger.info(buf3)
    pass
