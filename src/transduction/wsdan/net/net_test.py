import torch
from torch.nn import functional
import numpy as np
import cv2

from .metric import TopKAccuracyMetric
from .augment import batch_augment, get_raw_image, dump_heatmap

from ..shim import plot_attention, MriDataset, get_confusion_matrix, get_plt
plt = get_plt()

from ..demo.stats import softmax

import logging

from tqdm import tqdm
#from tqdm.notebook import tqdm


def get_scores(pred, true, class_names_sorted, debug=False):
    _score = 0

    predicted_cat = torch.tensor([], dtype=torch.long)
    labels_cat = torch.tensor([], dtype=torch.long)

    with np.printoptions(formatter={'float': '{:.2e}'.format}):
        for (i, (y_hat,y)) in enumerate(zip(pred,true)):
            _pred = torch.argmax(y_hat)
            _true = y

            predicted_cat = torch.cat((predicted_cat, torch.tensor([int(_pred)])), dim=0)
            labels_cat = torch.cat((labels_cat, torch.tensor([int(_true)])), dim=0)

            if debug:
                _pred_name = class_names_sorted[int(_pred)]
                _true_name = class_names_sorted[int(_true)]
                check = '✅' if _pred == _true else '❌'


                if _pred == _true:
                    _score += 1

                print("Case {}--{} {} (Pred={}, True={})".format(
                    i + 1, softmax(y_hat.cpu().numpy()), check, _pred_name, _true_name))

    if debug:
        print(f'@@ Accuracy: (# of ✅) / (# of Cases) = {_score} / {len(pred)} = %0.3f' % (_score / len(pred)))

    return labels_cat, predicted_cat


def overlay_bbox(im_in, bbox, imgW, imgH):
    scale_w = im_in.shape[0] / imgW
    scale_h = im_in.shape[1] / imgH
    return cv2.rectangle(
        cv2.cvtColor((im_in * 255.0).astype('uint8'), cv2.COLOR_RGB2BGR),
        (int(bbox[0] * scale_w), int(bbox[1] * scale_h)),
        (int(bbox[2] * scale_w), int(bbox[3] * scale_h)),
        (255, 0, 0), 2)

def verify_crop_attention(
    paths, ch, rh, savepath, ckpt_file,
    X, crop_images, crop_bboxes, attention_maps,
    y_true_scores, y_pred_scores):

    raw_image = get_raw_image(X.cpu())
    batches, _, imgH, imgW = X.size()

    for idx in range(batches):
        input_path = paths[idx]
        erica_mode = 'erica=' in input_path

        ytrue, ypred = y_true_scores[idx], y_pred_scores[idx]
        result = ytrue == ypred

        #---- input images
        bbox = crop_bboxes[idx]
        if erica_mode:
            im_input = plt.imread(input_path.split('?')[0])  # ndarray
            im_erica_l, im_erica_r = MriDataset.erica_crop_im(im_input, ch=ch, rh=rh)
            is_erica_left = 'erica=l' in input_path
            im_in = im_erica_l if is_erica_left else im_erica_r

            im_orig_out = overlay_bbox(im_in, bbox, imgW, imgH)
            if is_erica_left:
                im_erica_l = im_orig_out
            else:
                im_erica_r = im_orig_out

            li_input = [im_erica_l, im_erica_r]
        else:
            im_in = raw_image[idx].permute(1, 2, 0).numpy()
            li_input = [overlay_bbox(im_in, bbox, imgW, imgH)]

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
                 f'--------\n'
                 f'path: {input_path}\n'
                 f'model: {ckpt_file}\n'
                 f'ch: {ch} rh: {rh}\n'
                 f'window: {imgW}x{imgH} crop: {crop_bboxes[idx]}\n'
                 f'ytrue: {ytrue} ypred: {ypred} inference result: {result}')
        plot_attention(li_input + [im_crop_u8, im_heatmap], title,
            f'{savepath}/info_testds_{idx}_result_{result}.png')


def test(device, class_names_sorted, net, data_loader, ckpt, savepath=None,
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

            #---- ^^
            # @@ assume (data_loader's batch_size) == len(test_dataset)
            if i != 0:
                raise ValueError(f'@@ batch_size != len(test_dataset)')
            else:
                results_i_0 = (X, crop_images, y_pred, y, p)  # legacy compat

            if savepath is not None:
                y_true_scores, y_pred_scores = get_scores(
                    y_pred, y, class_names_sorted, debug=True)

                cm = get_confusion_matrix(y_true_scores, y_pred_scores, class_names_sorted)

                verify_crop_attention(
                    paths, ch, rh, savepath, ckpt,
                    X, crop_images, crop_bboxes, attention_maps,
                    y_true_scores, y_pred_scores)
            #---- $$

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

    return results_i_0, cm


def analyze_cm(cm, class_names=None, note=""):
    """
    Analyzes a confusion matrix using only NumPy.
    Calculates overall accuracy and per-class metrics.
    """
    cm = np.array(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        print("Error: Input must be a square 2D confusion matrix.")
        return None

    num_classes = cm.shape[0]
    if class_names is None or len(class_names) != num_classes:
        class_names = [f'Class {i+1}' for i in range(num_classes)]

    # 1. Overall Accuracy
    total_samples = np.sum(cm)
    tp_sum = np.trace(cm)
    overall_acc = tp_sum / total_samples if total_samples > 0 else 0

    print(note) # such as model name, epoch xx, with dopping etc.
    
    print(f"--- Overall Statistics ---")
    print(f"Total Samples: {total_samples}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({tp_sum}/{total_samples})\n")

    # 2. Per-Class Calculations (Vectorized)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = total_samples - (tp + fp + fn)
    actual_counts = tp + fn

    # Calculate metrics with division error handling
    recall = np.divide(tp, (tp + fn), out=np.full(num_classes, np.nan), where=(tp + fn) != 0)
    precision = np.divide(tp, (tp + fp), out=np.full(num_classes, np.nan), where=(tp + fp) != 0)
    
    # F1 Score: 2 * (P * R) / (P + R)
    f1_denom = precision + recall
    f1_score = np.divide(2 * precision * recall, f1_denom, 
                         out=np.full(num_classes, np.nan), 
                         where=(~np.isnan(f1_denom)) & (f1_denom > 0))
    
    sensitivity = np.divide(tp, (tp + fn), out=np.full(num_classes, np.nan), where=(tp + fn) != 0)
    specificity = np.divide(tn, (tn + fp), out=np.full(num_classes, np.nan), where=(tn + fp) != 0)

    # 3. Print Results in a formatted table style
    header = f"{'Class':<15} | {'Actual':<8} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<8}"
    print(header)
    print("-" * len(header))

    for i in range(num_classes):
        r = f"{recall[i]:.4f}" if not np.isnan(recall[i]) else "N/A"
        p = f"{precision[i]:.4f}" if not np.isnan(precision[i]) else "N/A"
        f1 = f"{f1_score[i]:.4f}" if not np.isnan(f1_score[i]) else "N/A"
        
        print(f"{class_names[i]:<15} | {int(actual_counts[i]):<8} | {r:<8} | {p:<10} | {f1:<8}")

    print("Sensitivity and Specificity per class:")
    print(f"{'Class':<15} | {'Sensitivity':<12} | {'Specificity':<12}")
    print("-" * 45)
    for i in range(num_classes):
        sens = f"{sensitivity[i]:.4f}" if not np.isnan(sensitivity[i]) else "N/A"
        spec = f"{specificity[i]:.4f}" if not np.isnan(specificity[i]) else "N/A"
        
        print(f"{class_names[i]:<15} | {sens:<12} | {spec:<12}")
    
