from settings import *
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.nn import Module
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from .training.consistency_losses import *
from .training.utils import setup_custom_logger
from .training.preprocessing import get_transformations
from .training.consistency_trainer import ModelTrainer
from .datasets.video_faces import VideoFacesDataset
from .datasets.data_utils import get_train, get_validation, get_test
from .training.evaluation import score_adience


def launch_all():
    start_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tensorboard_dir = os.path.join('runs', start_time + '_' + settings.EXPERIMENT_NAME)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger = setup_custom_logger(settings.LOG_NAME, tensorboard_dir + '/' + settings.EXPERIMENT_NAME + '.log')

    model = MODEL(**settings.MODEL_PARAMS)
    model = torch.nn.DataParallel(model)  # If training on subset of GPUs use: device_ids=[0, 1, 2, 3])
    model = model.to(settings.DEVICE)
    settings.ADAM_PARAMS['params'] = model.parameters()
    settings.SGD_PARAMS['params'] = model.parameters()
    optim, optim_params = (Adam(**settings.ADAM_PARAMS), settings.ADAM_PARAMS) if settings.OPTIM == 'adam' \
        else (SGD(**settings.SGD_PARAMS), settings.SGD_PARAMS)
    settings.SCHEDULER_PARAMS['optimizer'] = optim
    scheduler = ReduceLROnPlateau(**settings.SCHEDULER_PARAMS)

    params = {
        'model': model,
        'scheduler': scheduler,
        'image_loss': IMAGE_LOSS,
        'video_loss': settings.VIDEO_LOSS,
        'dataset_params': settings.DATASET_PARAMS,
        'use_vid_loss': settings.USE_VID_LOSS,
        'experiment_name': settings.EXPERIMENT_NAME,
        'epochs': settings.EPOCHS,
        'repickle': settings.REPICKLE,
        'keep': settings.KEEP,
        'batch_size': settings.BATCH_SIZE,
        'debug': settings.DEBUG,
        'initial_eval': settings.INITIAL_EVAL,
        'save_model': settings.SAVE_MODEL,
    }

    to_pickle = {**settings.MODEL_PARAMS, **optim_params, **settings.SCHEDULER_PARAMS, **params, **settings.DATASET_PARAMS}
    param_str = ''
    for k, v in sorted(to_pickle.items()):
        if isinstance(v, type(lambda x: True)):
            param_str += f"{str(k):<20}: {str(type(v)):<20} - source code: {str(inspect.getsource(v)):<40}\n"
            del to_pickle[k]
        elif isinstance(v, Module):
            param_str += f"{str(k):<20}: {str(type(v)):<20}\n"
            del to_pickle[k]
        elif isinstance(v, types.GeneratorType):
            param_str += f"{str(k):<20}: {str(type(v)):<20} - {str(v):<40}\n"
            del to_pickle[k]
        else:
            param_str += f"{str(k):<20}: {str(type(v)):<20} - {str(v):<40}\n"

    run_experiment(to_pickle, param_str, writer=writer, logger=logger, tensorboard_dir=tensorboard_dir, **params)


def run_experiment(to_pickle, param_str, model, scheduler, image_loss, video_loss,
                   dataset_params, writer, logger, tensorboard_dir, **kwargs):
    ###############################################################
    # Unpack arguments and throw error if extra keyword arguments #
    ###############################################################
    experiment_name = kwargs.pop('experiment_name', 'default')
    epochs = kwargs.pop('epochs', 0)
    repickle = kwargs.pop('repickle', True)
    keep = kwargs.pop('keep', lambda x: True)
    batch_size = kwargs.pop('batch_size', 64)
    debug = kwargs.pop('debug', False)
    use_vid_loss = kwargs.pop('use_vid_loss', lambda step, epoch: True)
    initial_eval = kwargs.pop('initial_eval', False)
    save_model = kwargs.pop('save_model', True)

    if len(kwargs) > 0:
        extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
        raise ValueError('Unrecognized arguments %s' % extra)

    ###############################################
    # Setup logging module and tensorboard writer #
    #           Pickle and log parameters         #
    ###############################################
    save_path = tensorboard_dir + '/' + experiment_name + '.pth'
    pickle.dump(to_pickle, open(tensorboard_dir + '/' + experiment_name + ".pickle", "wb"))
    logger.info(param_str)

    fname = '.pickle'

    #####################
    # Setup dataloaders #
    #####################
    trans = get_transformations()

    if repickle:
        tr = get_train(dataset_params['train'], trans[0])
        tr, val = get_validation(dataset_params['val'], trans[1], to_split=tr)
        ts = get_test(dataset_params['test'], trans[1])
        with open('tr' + fname, 'wb') as f:
            pickle.dump(tr, f)
        with open('val' + fname, 'wb') as f:
            pickle.dump(val, f)
        with open('ts' + fname, 'wb') as f:
            pickle.dump(ts, f)
    else:
        logger.info(f"Using existing training pickle at {fname}")
        tr = pickle.load(open('tr' + fname, "rb"))
        val = pickle.load(open('val' + fname, "rb"))
        ts = pickle.load(open('ts' + fname, "rb"))

    tr = DataLoader(tr, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=24 if not debug else 0)
    val = DataLoader(val, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8 if not debug else 0)
    ts = DataLoader(ts, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8 if not debug else 0)

    if dataset_params['video'] is None or dataset_params['video'] > 0:
        vids = VideoFacesDataset(transform=trans[0], batch_size=batch_size, subsample=dataset_params['video'])
        vids_loader = DataLoader(vids, pin_memory=True, batch_sampler=vids.sampler,
                                 num_workers=8 if not debug else 0)
    else:
        vids_loader = None

    # Start Experiments
    trainer = ModelTrainer(model, image_loss, video_loss, scheduler, writer=writer, save_path=save_path)

    if initial_eval:
        init_mae, init_loss = trainer.score(val, to_log=10, epoch=0)
        logger.info(f"Initial VAL MAE: {init_mae:.3f}")
        if vids_loader:
            trainer.score_videos(vids_loader, to_log=10, epoch=0)

    if epochs > 0:
        trainer.fit(tr, val, vids_loader, epochs=epochs, use_vid_loss=use_vid_loss, save=save_model)

    score_adience(model, writer, image_loss, subset='test')

    final_mae, final_loss = trainer.score(ts, fold='test', epoch=epochs+1)
    logger.info(f"TEST MAE: {final_mae:.3f}")

    writer.export_scalars_to_json("./runs/all_scalars.json")
    writer.close()  # Close tensorboard writer
