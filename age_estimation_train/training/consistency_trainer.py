import math
from itertools import cycle

from torch.optim.lr_scheduler import MultiStepLR

import settings
from .evaluation import *
from .tensorboard import umd_faces_log_figs
from .utils import catch_stdin


class ModelTrainer:

    def __init__(self, model, image_loss, video_loss, scheduler, writer=None, save_path=None, validator=None):
        self.frames_per_vid = FRAMES_PER_VID
        self.logger = logging.getLogger(LOG_NAME)
        self.writer = writer
        self.save_path = save_path

        self.model = model
        self.image_loss = image_loss
        self.video_loss = video_loss
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer
        self.validator = validator
        self.video_iter = None

    def fit(self, train, validate, aux=None, epochs=1, save=True, use_vid_loss=lambda s, e: True):
        model = self.model
        model.train()

        self.video_iter = cycle(aux) if aux else None
        high_score = 0
        step = 0

        for epoch in range(1, epochs + 1):

            if isinstance(self.scheduler, MultiStepLR) and epoch in self.scheduler.milestones:
                self.logger.info(f"Decreasing learning rate by factor {self.scheduler.gamma}")
                for param_group in self.optimizer.param_groups:
                    self.logger.info(f"Learning rate: {param_group['lr']}")
            val_loss = None
            tr_loss = 0
            count = 0
            for inputs, labels in train:
                gt = get_label(self.image_loss, labels)
                inputs = inputs.to(settings.DEVICE)
                outputs = model.forward(inputs)
                image_loss = self.image_loss(outputs, gt)
                self.optimizer.zero_grad()
                image_loss.backward()
                self.optimizer.step()

                if aux and use_vid_loss(step, epoch):
                    frames, paths, vid = next(self.video_iter)
                    frames = frames.to(settings.DEVICE)
                    out_frames = model.forward(frames)
                    video_loss = self.video_loss(out_frames, epoch)
                    self.optimizer.zero_grad()
                    video_loss.backward()
                    self.optimizer.step()
                    loss = image_loss + video_loss
                else:
                    loss = image_loss
                    video_loss = 0

                if math.isnan(loss) or loss > 10000:
                    self.logger.critical(f"Loss explosion {loss} -- aborting")
                    return

                self.writer.add_scalars('train/loss',
                                        {'_IMAGE_LOSS': image_loss, '_VIDEO_LOSS': video_loss,
                                         'total_loss': loss}, step)
                tr_loss += loss.detach()
                step += 1
                count += 1
            # Finish loop
            tr_loss = tr_loss / count
            self.logger.info(f"Finished Epoch {epoch} with tr_loss {tr_loss}")

            if self.validator is not None:
                score = self.validator.validate(self.model)
                preds = {'tp': score['tp'], 'fp': score['fp'], 'tn': score['tn'], 'fn': score['fn']}
                stats = {'acc': score['acc'], 'f1': score['f1'], 'prec': score['prec'], 'rec': score['rec']}

                if save and score['f1'] > high_score:
                    torch.save(model.state_dict(), self.save_path)
                    high_score = score['f1']
                elif save:
                    torch.save(model.state_dict(), self.save_path + 'recent')

                self.writer.add_scalars('preds', preds, epoch)
                self.writer.add_scalars('stats', stats, epoch)
                self.writer.add_scalar('val_loss', score['loss'], epoch)
            elif validate is not None:
                #score_adience(model, self.writer, self.image_loss, tag=str(epoch))
                #score_appa_real(model, self.writer, self.image_loss, tag=str(epoch))
                if epoch % 20 == 0:  # Every 5 _EPOCHS get plots to visualize distributions
                    to_log = 16
                else:
                    to_log = 0
                
                score, val_loss = self.score(validate, to_log=to_log, fold='val', epoch=epoch)

                self.logger.info(f"VAL Epoch {epoch} MAE - {score:.3f}")
                self.writer.add_scalars('train/loss',
                                        {'val_image_loss': val_loss}, step)
                if save and score < high_score:
                    torch.save(model.state_dict(), self.save_path)
                    high_score = score
                elif save:
                    torch.save(model.state_dict(), self.save_path + 'recent')

            if epoch % 1000 == 0:  # Every 5 _EPOCHS, check MAE on train
                score, train_loss = self.score(train, to_log=32, fold='train', epoch=epoch)
                self.logger.info(f"TRAIN Epoch {epoch} MAE - {score:.3f}")
                self.writer.add_scalars('train/loss',
                                        {'train_image_loss': train_loss}, step)

            if aux:
                self.score_videos(aux, to_log=8, epoch=epoch)
            self.logger.info(f"Val_loss: {val_loss}")
            msg = catch_stdin(self.scheduler.step, tr_loss)
            if msg:
                self.logger.warning(f"Scheduler says {msg}")
        return

    def predict(self, x):
        """ Forward pass with no gradients. """
        self.model.eval()
        with torch.no_grad():
            x = x.to(settings.DEVICE)
            outputs = self.model.forward(x)
            pred = torch.matmul(torch.exp(outputs), settings.CLASSES).cpu().numpy()
        return pred.cpu().numpy()

    def score(self, data: DataLoader, to_log=-1, fold='val', epoch=None):
        """ Score a model on the given data. """
        self.model.eval()
        mae = 0
        mae_count = 0

        loss = 0
        loss_count = 0
        with torch.no_grad():
            for inputs, labels in data:
                gt = get_label(self.image_loss, labels)
                t = get_label(NLLLoss(), labels).to(torch.float32)

                inputs = inputs.to(settings.DEVICE)
                outputs = self.model.forward(inputs)
                loss += self.image_loss(outputs, gt)

                pred = outputs[:, 0]
                '''
                if isinstance(outputs, tuple):
                    dist = outputs[0]
                else:
                    dist = outputs

                pred = torch.matmul(torch.exp(dist), settings.CLASSES).view(-1)
                '''

                if to_log > 0:
                    if data.batch_size - to_log > 0:
                        idx = to_log
                    else:
                        idx = data.batch_size
                    l = {k: v for i, (k, v) in enumerate(labels.items()) if i < idx}
                    appa_real_log_figs(dist[:idx], pred[:idx], l, self.writer, fold=fold, epoch=epoch)
                    to_log -= idx

                mae += (t - pred).abs().sum()
                mae_count += len(pred)
                loss_count += 1
        loss /= max(loss_count, 1)

        score = (mae/max(mae_count, 1)).type(torch.FloatTensor).item()
        self.writer.add_scalar(f'{fold}/mae', score, epoch)
        self.writer.add_scalar(f'{fold}/loss', loss, epoch)

        return score, loss

    def score_videos(self, aux: DataLoader, to_log=8, epoch=None):
        self.model.eval()
        aux = iter(aux)
        counter = 0
        with torch.no_grad():
            while counter < to_log:
                frames, paths, vid = next(aux)
                frames = frames.to(settings.DEVICE)
                out_frames = self.model.forward(frames)
                video_ages = torch.matmul(torch.exp(out_frames), settings.CLASSES).view(-1)

                split_out = out_frames.split(self.frames_per_vid)
                split_ages = video_ages.split(self.frames_per_vid)
                split_paths = [paths[x:x + self.frames_per_vid] for x in range(0, len(paths), self.frames_per_vid)]
                for j in range(len(split_out)):
                    if counter >= to_log:
                        break
                    umd_faces_log_figs(split_out[j], split_ages[j], split_paths[j], self.writer, epoch=epoch)
                    counter += 1
