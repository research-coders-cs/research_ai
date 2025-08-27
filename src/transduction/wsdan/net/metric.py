import numpy as np


class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples
