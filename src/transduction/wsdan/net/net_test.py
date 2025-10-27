import torch
from torch.nn import functional
import numpy as np

from .metric import TopKAccuracyMetric
from .augment import batch_augment, get_raw_image, dump_heatmap

from ..shim import get_scores, plot_attention, MriDataset, get_plt
plt = get_plt()

import logging

from tqdm import tqdm
#from tqdm.notebook import tqdm


def test(device, net, data_loader, ckpt, savepath=None,
         ch=None, rh=None):
    logging.info('Network loading from {}'.format(ckpt))

    ckpt_dict = torch.load(ckpt, weights_only=False)

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
            crop_images, crop_bboxes = batch_augment(X, paths, attention_maps,
                savepath=None, use_doppler=False,
                mode='crop', theta=0.85, padding_ratio=0.05)

            # crop images forward
            y_pred_crop, _, _ = net(crop_images)
            importance =  torch.abs(y_pred_raw[0] - y_pred_raw[1])

            y_pred = (y_pred_raw + (y_pred_crop * 2 * importance)) / 3.

            #-------- ^^
            # @@ assume (data_loader's batch_size) == len(test_dataset)
            if i != 0:
                raise ValueError(f'@@ batch_size != len(test_dataset)')
            else:
                print('@@ crop_bboxes:', crop_bboxes)
                results_i_0 = (X, crop_images, y_pred, y, p)

            if savepath is not None:
                raw_image = get_raw_image(X.cpu())
                batches, _, imgH, imgW = X.size()

                # TODO refactor into API
                ckpt_file = 'N/A'  # !!!!
                class_names_sorted = ['E0', 'E1', 'E2', 'E3']  # !!!!

                y_true_scores, y_pred_scores = get_scores(results_i_0, class_names_sorted)

                for idx in range(batches):
                    input_path = paths[idx]
                    erica_mode = 'erica=' in input_path

                    ytrue, ypred = y_true_scores[idx], y_pred_scores[idx]
                    result = ytrue == ypred

                    #---- input images
                    if erica_mode:
                        im_input = plt.imread(input_path.split('?')[0])  # ndarray
                        im_erica_l, im_erica_r = MriDataset.erica_crop_im(im_input, ch=ch, rh=rh)
                    else:
                        im_orig = raw_image[idx].permute(1, 2, 0).numpy()
                    li_input = [im_erica_l, im_erica_r] if erica_mode else [im_orig]

                    #---- crop image
                    crop_images_cpu = crop_images.cpu()
                    im_crop = crop_images_cpu[idx].permute(1, 2, 0).numpy()
                    array_min = im_crop.min()
                    array_max = im_crop.max()
                    normalized_array = (im_crop - array_min) / (array_max - array_min)  # Normalize to [0, 1] first
                    im_crop_u8 = (normalized_array * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

                    #---- heatmap image
                    im_heatmap = dump_heatmap(
                        savepath, '%06d' % idx,
                        raw_image, attention_maps[idx:idx + 1], imgH, imgW, idx)


                    title = (f'testds[{idx}]: input | crop | attention\n'
                             f'(ch: {ch} rh: {rh})\n'
                             f'(path: {input_path})\n'
                             f'(ytrue: {ytrue} ypred: {ypred} inference result: {result})\n'
                             f'(model: {ckpt_file})')
                    plot_attention(li_input + [im_crop_u8, im_heatmap], title,
                        f'{savepath}/info_testds_{idx}_result_{result}.png')

                    #raise ValueError('!!!! debug !!!!')
            #-------- $$

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

    return results_i_0
