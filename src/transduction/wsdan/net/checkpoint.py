import torch
import numpy as np


class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max', savemode_debug=False):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.savemode_debug = savemode_debug
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, num_epoch, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]
        savepath = self.get_savepath_last()

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score
            savepath = self.get_savepath_last()

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, savepath)
            print(f'@@ [ckpt:UPDATED] epoch: {num_epoch} best: %.3f savepath: {savepath}' % self.best_score)
        else:
            print(f'@@ [ckpt:unchanged] epoch: {num_epoch} best (current): %.3f (%.3f) savepath: {savepath}' % (self.best_score, current_score))

    def get_savepath_last(self):
        return self.savepath + ("_%.3f" % self.best_score if self.savemode_debug else '')
