import torch
from torch.nn import functional

from .metric import TopKAccuracyMetric
from .augment import batch_augment, get_raw_image, dump_heatmap

import logging

from tqdm import tqdm
#from tqdm.notebook import tqdm


def test(device, net, batch_size, data_loader, ckpt, savepath=None):
    logging.info('Network loading from {}'.format(ckpt))

    ckpt_dict = torch.load(ckpt)

    print('@@ ckpt:', ckpt)
    for key, val in ckpt_dict.items():  # @@
        print('@@ ckpt_dict - key:', key)
        if key == 'logs': print('  ', val)
        if key == 'feature_center': print('  ', val)

    state_dict = ckpt_dict['state_dict']
    # for key, _ in state_dict.items(): print('@@ state_dict - key:', key)  # @@

    net.load_state_dict(state_dict)
    #exit()  # @@ !!

    raw_accuracy = TopKAccuracyMetric()
    ref_accuracy = TopKAccuracyMetric()
    raw_accuracy.reset()

    results = []
    net.eval()

    with torch.no_grad():

        pbar = tqdm(total=len(data_loader), unit=' batches')
        pbar.set_description('Test data')

        for i, (X, y, p) in enumerate(data_loader):
            paths = p['path']  # @@

            # obtain data for testing
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, _, attention_maps = net(X)

            ##################################
            # Attention Cropping
            ##################################
            crop_image = batch_augment(X, paths, attention_maps,
                savepath=None, use_doppler=False,
                mode='crop', theta=0.85, padding_ratio=0.05)

            # crop images forward
            y_pred_crop, _, _ = net(crop_image)
            importance =  torch.abs(y_pred_raw[0] - y_pred_raw[1])

            y_pred = (y_pred_raw + (y_pred_crop * 2 * importance)) / 3.

            if savepath is not None:
                raw_image = get_raw_image(X.cpu())
                batches, _, imgH, imgW = X.size()
                for idx in range(batches):
                    dump_heatmap(savepath, '%06d' % (i * batch_size + idx),
                                 raw_image, attention_maps[idx:idx + 1], imgH, imgW, idx)

            results = (X, crop_image, y_pred, y, p)

            # Top K
            epoch_raw_acc = raw_accuracy(y_pred_raw, y)
            epoch_ref_acc = ref_accuracy(y_pred, y)

            # end of this batch
            batch_info = 'Val Acc: Raw ({:.2f}), Refine ({:.2f})'.format(
                epoch_raw_acc[0], epoch_ref_acc[0])
            pbar.update()
            pbar.set_postfix_str(batch_info)
            torch.cuda.empty_cache()

        pbar.close()

    return results