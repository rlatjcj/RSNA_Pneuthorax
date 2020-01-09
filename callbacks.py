import tqdm
import numpy as np
import csv
import six
from collections import OrderedDict
from collections import Iterable

import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler

class Evaluate(keras.callbacks.Callback):
    def __init__(
        self,
        generator,
        eval_steps,
        is_patch,
        verbose=1):

        super(Evaluate, self).__init__()
        self.generator = generator
        self.eval_steps = eval_steps
        self.is_patch = is_patch
        self.verbose = verbose

        self.tot_dice = 0.
        self.pneu_dice = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            logs = logs or {}
            self.tot_dice, self.pneu_dice = self.evaluate()

            if self.verbose == 1:
                print('Epoch {:4d} - Dice score : {:.4f} | Pneu score : {:.4f}'.format(epoch+1, self.tot_dice, self.pneu_dice))
            
        logs['eval_dice'] = self.tot_dice
        logs['eval_dice_pneu'] = self.pneu_dice

    def dice(self, y_true, y_pred, classes, smooth=1.):
        loss = 0.   
        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        if classes > 1:
            for num_label in range(classes):
                y_true_f = y_true[...,num_label].flatten()
                y_pred_f = y_pred[...,num_label].flatten()
                intersection = np.sum(y_true_f * y_pred_f)
                loss += (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

        else:
            y_true_f = y_true.flatten()
            y_pred_f = y_pred.flatten()
            intersection = np.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

        return loss / classes

    def onehot(self, x, classes):
        result = np.zeros(x.shape+(classes,))
        for i in range(classes):
            result[...,i][np.where(x == i)] = 1.
        return result

    def evaluate(self, classes=2):
        tot_dice = []
        pneu_dice = []
        for i in tqdm.trange(self.eval_steps):
            y_true = np.zeros((1024, 1024, 2))
            y_pred = np.zeros((1024, 1024, 2))
            loc_cnt = np.zeros((1024, 1024, 2))

            g = next(self.generator)
            result = self.model.predict_on_batch(g[0])

            if self.is_patch:
                ratio = 1024//result.shape[1]
                for l in range(result.shape[0]):
                    # result = model.predict_on_batch(g[0][l][np.newaxis,...])
                    y_true[(l//(2*ratio-1))*1024//(ratio*2):(l//(2*ratio-1)+2)*1024//(ratio*2), 
                           (l%(2*ratio-1))*1024//(ratio*2):(l%(2*ratio-1)+2)*1024//(ratio*2)] += g[1][l]

                    y_pred[(l//(2*ratio-1))*1024//(ratio*2):(l//(2*ratio-1)+2)*1024//(ratio*2), 
                           (l%(2*ratio-1))*1024//(ratio*2):(l%(2*ratio-1)+2)*1024//(ratio*2)] += result[l]

                    loc_cnt[(l//(2*ratio-1))*1024//(ratio*2):(l//(2*ratio-1)+2)*1024//(ratio*2), 
                            (l%(2*ratio-1))*1024//(ratio*2):(l%(2*ratio-1)+2)*1024//(ratio*2)] += 1
                
                y_true /= loc_cnt
                y_pred /= loc_cnt

            else:
                y_pred = result

            y_pred = self.onehot(np.argmax(y_pred, axis=-1), classes)                
            tot_dice.append(self.dice(y_true, y_pred, classes))
            pneu_dice.append(self.dice(y_true[...,1], y_pred[...,1], 1))
        
        return sum(tot_dice)/eval_steps, sum(pneu_dice)/eval_steps


class CustomBatchLogger(CSVLogger):
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator, append)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last batch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['batch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        super().on_train_end(logs)


def callback_checkpoint(filepath, monitor, verbose, mode, save_best_only, save_weights_only):
    return ModelCheckpoint(filepath=filepath,
                           monitor=monitor,
                           verbose=verbose,
                           mode=mode,
                           save_best_only=save_best_only,
                           save_weights_only=save_weights_only)

def callback_epochlogger(filename, separator, append):
    return CSVLogger(filename=filename,
                     separator=separator,
                     append=append)

def callback_batchlogger(filename, separator, append):
    return CustomBatchLogger(filename=filename,
                             separator=separator,
                             append=append)

def callback_tensorboard(log_dir, batch_size):
    return TensorBoard(log_dir=log_dir, batch_size=batch_size)

def main_schedule(initlr, mode, value, duration, total_epoch):
    def _cosine_anneal_schedule(epoch, total_epoch):
        total_epoch = np.maximum(total_epoch, 50)
        cos_inner = np.pi * (epoch % total_epoch)
        cos_inner /= total_epoch
        cos_out = np.cos(cos_inner) + 1
        return float(initlr / 2 * cos_out)

    def _exponential(epoch, value, duration):
        update = epoch // duration
        return initlr * (value ** update)

    def _schedule(epoch, lr):
        if mode == 'exponential':
            return _exponential(epoch, value, duration)
        elif mode == 'cosine':
            return _cosine_anneal_schedule(epoch, total_epoch)

    return _schedule

def callback_learningrate(initlr, mode, value, duration, total_epoch):
    return LearningRateScheduler(schedule=main_schedule(initlr=initlr, 
                                                        mode=mode, 
                                                        value=value, 
                                                        duration=duration, 
                                                        total_epoch=total_epoch), verbose=1)

def callback_evaluate(generator, eval_steps, is_patch):
    return Evaluate(generator=generator, eval_steps=eval_steps, is_patch=is_patch, verbose=1)