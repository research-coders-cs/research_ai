import os
import cv2
import numpy as np
import pandas as pd
import hashlib

import logging
logger = logging.getLogger('@@')


def detect_doppler(img):
    if len(img.shape) < 3:
        #print('gray_scale')
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    green_image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    green_image[:,:] = img[:,:,1]
    _, threshold = cv2.threshold(green_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imwrite('check.jpg', threshold)

    cntrRect = None
    maxarea = 0
    for cnt in contours:
        hull = cv2.convexHull(cnt,returnPoints = True)
        #print(cnt)
        #print(hull)
        approx = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(cnt, True), True)
        #cv2.drawContours(img, [approx], 0, (0), 5)
        if len(approx) == 4:
            xy = approx.ravel()
            area = cv2.contourArea(cnt)
            if maxarea < area:
                use = True
                for i in range(4):
                    lx1 = xy[i*2]
                    ly1 = xy[i*2+1]
                    if i == 3:
                        lx2 = xy[0]
                        ly2 = xy[1]
                    else:
                        lx2 = xy[(i+1)*2]
                        ly2 = xy[(i+1)*2+1]
                    angle = np.absolute(np.arctan2(ly2-ly1, lx2-lx1)) * 180 / np.pi
                    if angle > 45 and angle < 135:
                        angle = angle - 90
                    if angle >= 135:
                        angle = angle - 180
                    if angle > 3:
                        use = False
                        break

                if use:
                    maxarea = area
                    cntrRect = approx

    if cntrRect is None:
        return None

    xy = cntrRect.ravel()
    x = [xy[0], xy[2], xy[4], xy[6]]
    y = [xy[1], xy[3], xy[5], xy[7]]
    min_x = float(min(x))
    max_x = float(max(x))
    min_y = float(min(y))
    max_y = float(max(y))

    return [min_x, min_y, max_x, max_y]


def get_iou(truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(truth[0], pred[0])
    iy1 = np.maximum(truth[1], pred[1])
    ix2 = np.minimum(truth[2], pred[2])
    iy2 = np.minimum(truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width
    # print("area_of_intersection : ", area_of_intersection)

    w_mark =  (pred[2] - pred[0])
    h_mark = (pred[3] - pred[1])
    area_mark = w_mark * h_mark
    # print("area_mark : ", area_mark)

    # Ground Truth dimensions.
    gt_height = truth[3] - truth[1] + 1
    gt_width = truth[2] - truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    # print("area_of_union : ", area_of_union)

    _iou = area_of_intersection / area_of_union
    #@@ iou = np.round(_iou * 100, 2)

    _intersection_of_mark = area_of_intersection / area_mark
    #@@ intersection_of_mark = np.round(_intersection_of_mark * 100, 2)

    logger.debug('get_iou(): area_{of_intersection,mark,of_union}, intersection_of_mark: %0.1f, %0.1f, %0.1f' % (
        area_of_intersection, area_mark, area_of_union))

    #@@return iou, intersection_of_mark
    return _iou, _intersection_of_mark

#

def doppler_comp(path_doppler, path_markers, path_markers_label):
    img_doppler = cv2.imread(path_doppler)
    width = int(img_doppler.shape[1])
    height = int(img_doppler.shape[0])

    temp = detect_doppler(img_doppler)

    x1_doppler_calc = int(temp[0])
    y1_doppler_calc = int(temp[1])
    x2_doppler_calc = int(temp[2])
    y2_doppler_calc = int(temp[3])

    bbox_doppler = np.array([
        x1_doppler_calc, y1_doppler_calc, x2_doppler_calc, y2_doppler_calc],
        dtype=np.float32)
    border_img_doppler = cv2.rectangle(img_doppler, (x1_doppler_calc, y1_doppler_calc), (x2_doppler_calc, y2_doppler_calc), (255, 255, 0), 2)

    #

    label_markers = pd.read_csv(path_markers_label, header=None, index_col=False).to_numpy()
    temp_row = []
    for row in label_markers:
        temp_col = []
        for col in row:
            for item in col.split():
                temp_col.append(float(item))
        temp_row.append(np.array(temp_col))
    temp = np.array(temp_row)

    x1_markers = np.min(temp[:,1])
    x2_markers = np.max(temp[:,1])
    y1_markers = np.min(temp[:,2])
    y2_markers = np.max(temp[:,2])

    x1_markers_calc = int(width * x1_markers)
    y1_markers_calc = int(height * y1_markers)
    x2_markers_calc = int(width * x2_markers)
    y2_markers_calc = int(height * y2_markers)
    bbox_markers = np.array(
        [x1_markers_calc, y1_markers_calc, x2_markers_calc, y2_markers_calc],
        dtype=np.float32)

    #

    img_markers = cv2.imread(path_markers)

    unborder_img_markers = cv2.resize(img_markers, (width, height))
    unborder_img_markers = cv2.rectangle(
        unborder_img_markers,
        (int(width * x1_doppler_calc), int(height * y1_doppler_calc)),
        (int(width * x2_doppler_calc), int(height * y2_doppler_calc)),
        (255, 255, 0), 2)
    border_img_markers = cv2.rectangle(
        unborder_img_markers.copy(),
        (int(width * x1_markers), int(height * y1_markers)),
        (int(width * x2_markers), int(height * y2_markers)),
        (255, 0, 0), 2)

    # doppler
    border_img_markers = cv2.rectangle(
        border_img_markers,
        (x1_doppler_calc, y1_doppler_calc),
        (x2_doppler_calc, y2_doppler_calc),
        (255, 255, 0), 2)

    return bbox_doppler, bbox_markers, border_img_doppler, border_img_markers


def plot_comp(border_img_doppler, border_img_markers, path_doppler, path_markers):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
    for idx, (title, border_img) in enumerate((
            (f'Doppler : {os.path.basename(path_doppler)}', border_img_doppler),
            (f'Markers : {os.path.basename(path_markers)}', border_img_markers))):
        ax[idx].title.set_text(title)
        ax[idx].imshow(border_img)
        ax[idx].grid(False)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])

    return plt


def get_sample_paths():  # used by `demo_doppler_compare()`
    return (
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0011_1_p0022.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0011_2_p0022.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0011_2_p0022.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0076_1_p0152.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0076_2_p0152.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0076_2_p0152.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0022_1_p0044.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0022_2_p0044.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0022_2_p0044.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule3_0001-0030_c0024_2_p0071.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule3_0001-0030_c0024_1_p0071.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule3_0001-0030_c0024_1_p0071.txt'),
        # !! removed jpg causing issues on data_loader
        # ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule2_0001-0016_c0001_3_p0002.jpg',
        # './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule2_0001-0016_c0001_1_p0002.png',
        # './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule2_0001-0016_c0001_1_p0002.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_siriraj_0001-0160_c0128_2_p0089.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_siriraj_0001-0160_c0128_1_p0088.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_siriraj_0001-0160_c0128_1_p0088.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0008_1_p0016.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0008_2_p0016.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Benign/benign_nodule1_0001-0100_c0008_2_p0016.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_siriraj_0001-0124_c0110_3_p0257.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.txt'),
        ('./Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_nodule3_0001-0030_c0004_3_p0011.png',
        './Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.png',
        './Siriraj_sample_doppler_comp/Markers_Train_Markers_Labels/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.txt'),
    )

#

def get_to_doppler(root):

    dir_mtrm_benign = f'{root}/Markers_Train_Remove_Markers/Benign_Remove/train'
    dir_mtrm_malignant = f'{root}/Markers_Train_Remove_Markers/Malignant_Remove/train'
    dir_dtc_benign = f'{root}/Doppler_Train_Crop/Benign/matched'
    dir_dtc_malignant = f'{root}/Doppler_Train_Crop/Malignant/matched'

    dir_mtrm_validate_benign = f'{root}/Markers_Train_Remove_Markers/Benign_Remove/validate'
    dir_mtrm_validate_malignant = f'{root}/Markers_Train_Remove_Markers/Malignant_Remove/validate'
    dir_dtc_validate_benign = f'{root}/Doppler_Train_Crop/Benign_Remove/validate_synthesized'
    dir_dtc_validate_malignant = f'{root}/Doppler_Train_Crop/Malignant_Remove/validate_synthesized'

    to_doppler = {
        #-------- 'Siriraj_sample_doppler_comp'
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0011_2_p0022.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0011_1_p0022.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0076_2_p0152.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0076_1_p0152.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0022_2_p0044.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0022_1_p0044.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule3_0001-0030_c0024_1_p0071.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule3_0001-0030_c0024_2_p0071.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_siriraj_0001-0160_c0128_1_p0088.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_siriraj_0001-0160_c0128_2_p0089.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Benign/benign_nodule1_0001-0100_c0008_2_p0016.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Benign/benign_nodule1_0001-0100_c0008_1_p0016.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_siriraj_0001-0124_c0110_2_p0256.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_siriraj_0001-0124_c0110_3_p0257.png',
        'Siriraj_sample_doppler_comp/Markers_Train/Malignant/malignant_nodule3_0001-0030_c0004_1_p0011.png':
        'Siriraj_sample_doppler_comp/Doppler_Train_Crop/Malignant/malignant_nodule3_0001-0030_c0004_3_p0011.png',
        #-------- via _20
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0034_2_p0068.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0034_1_p0068.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0061_2_p0122.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0061_1_p0122.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0066_2_p0132.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0066_1_p0132.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0090_2_p0180.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0090_1_p0180.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0001_1_p0002.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0001_2_p0002.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0002_1_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0002_2_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0008_2_p0023.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0008_3_p0023.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0013_1_p0038.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0013_3_p0038.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0024_1_p0071.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0024_2_p0071.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0025_1_p0074.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0025_2_p0074.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0012_1_p0035.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0012_2_p0035.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0014_2_p0041.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0014_4_p0041.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0021_1_p0062.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0021_3_p0062.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0033_1_p0008.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0033_3_p0008.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0057_1_p0080.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0057_2_p0080.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0046_1_p0047.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0046_2_p0047.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0047_1_p0050.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0047_3_p0050.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0050_1_p0059.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0050_2_p0059.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0051_1_p0062.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0051_3_p0062.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0053_1_p0068.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0053_2_p0068.png',
        #-------- via _100a
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0107_2_p0014.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0007_2_p0014.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0122_1_p0044.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0022_1_p0044.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0135_1_p0070.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0035_1_p0070.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0140_1_p0080.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0040_1_p0080.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0001_2_p0002.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0001_4_p0002.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0002_1_p0005.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0002_3_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0008_2_p0023.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0008_3_p0023.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0003_1_p0008.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0003_2_p0008.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0009_2_p0026.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0009_3_p0026.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0010_1_p0029.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0010_4_p0029.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0004_2_p0011.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0004_4_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0011_2_p0032.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0011_3_p0032.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0012_2_p0035.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0012_4_p0035.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0005_2_p0014.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0005_4_p0014.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0014_2_p0038.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0014_3_p0038.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0006_2_p0017.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0006_3_p0017.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0015_2_p0041.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0015_3_p0041.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0007_2_p0020.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0007_4_p0020.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0004_2_p0012.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0004_3_p0013.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0132_1_p0100.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0132_2_p0101.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0133_1_p0103.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0133_2_p0104.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0134_1_p0106.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0134_2_p0107.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0136_1_p0111.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0136_2_p0112.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0137_1_p0114.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0137_2_p0115.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0022_1_p0029.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0022_2_p0030.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0138_1_p0117.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0138_2_p0118.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0023_2_p0033.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0023_3_p0034.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0139_1_p0120.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0139_2_p0121.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0024_1_p0036.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0024_2_p0037.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0140_1_p0123.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0140_1_p0124.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0025_2_p0040.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0025_3_p0041.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0141_1_p0126.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0141_2_p0127.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0026_2_p0044.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0026_3_p0045.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0142_1_p0129.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0142_2_p0130.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0027_1_p0047.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0027_2_p0048.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0143_1_p0132.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0143_2_p0133.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0028_2_p0051.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0028_3_p0052.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0029_2_p0055.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0029_3_p0056.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0030_1_p0058.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0030_2_p0059.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0146_1_p0139.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0146_2_p0140.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0031_1_p0061.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0031_2_p0062.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0147_1_p0142.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0147_2_p0143.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0035_2_p0065.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0035_3_p0066.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0148_1_p0145.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0148_2_p0146.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0149_1_p0148.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0149_2_p0149.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0150_1_p0151.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0150_2_p0152.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0066_1_p0072.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0066_2_p0073.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0151_2_p0155.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0151_2_p0156.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0152_1_p0158.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0152_2_p0159.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0153_1_p0161.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0153_2_p0162.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0125_1_p0079.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0125_2_p0080.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0154_1_p0164.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0154_2_p0165.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0155_1_p0167.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0155_2_p0168.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0127_2_p0085.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0127_3_p0086.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0156_1_p0170.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0156_2_p0171.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0128_1_p0088.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0128_2_p0089.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0158_1_p0176.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0158_2_p0177.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0129_1_p0091.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0129_2_p0092.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0159_1_p0179.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0159_2_p0180.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0160_1_p0182.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0160_2_p0183.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0131_1_p0097.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0131_2_p0098.png',
        f'{dir_mtrm_malignant}/malignant_nodule2_c0013_1_p0003.png': f'{dir_dtc_malignant}/malignant_nodule2_c0013_3_p0003.png',
        f'{dir_mtrm_malignant}/malignant_nodule2_c0018_1_p0009.png': f'{dir_dtc_malignant}/malignant_nodule2_c0018_4_p0009.png',
        f'{dir_mtrm_malignant}/malignant_nodule2_c0017_1_p0006.png': f'{dir_dtc_malignant}/malignant_nodule2_c0017_4_p0006.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0088_1_p0017.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0088_2_p0017.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0111_2_p0063.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0111_3_p0063.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0089_1_p0019.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0089_2_p0019.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0112_2_p0065.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0112_3_p0065.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0091_2_p0023.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0091_3_p0023.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0092_1_p0025.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0092_2_p0025.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0093_1_p0027.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0093_3_p0027.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0118_2_p0077.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0118_3_p0077.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0096_1_p0033.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0096_2_p0033.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0119_1_p0079.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0119_3_p0079.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0121_1_p0083.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0121_3_p0083.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0104_2_p0049.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0104_3_p0049.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0123_1_p0087.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0123_2_p0087.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0005_1_p0003.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0005_2_p0004.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0069_2_p0127.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0069_3_p0128.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0007_1_p0009.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0007_2_p0010.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0008_1_p0012.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0008_2_p0013.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0076_1_p0150.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0076_2_p0151.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0011_1_p0022.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0011_2_p0023.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0080_1_p0162.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0080_2_p0163.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0081_2_p0166.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0081_3_p0167.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0082_1_p0169.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0082_2_p0170.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0014_2_p0028.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0014_3_p0029.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0087_1_p0180.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0087_2_p0181.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c001x_2_p0137.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c001x_3_p0138.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0095_1_p0207.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0095_2_p0208.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0097_2_p0215.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0097_1_p0216.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0098_1_p0218.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0098_2_p0219.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0099_1_p0221.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0099_2_p0222.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0101_2_p0227.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0101_3_p0228.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0044_1_p0064.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0044_2_p0065.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0105_1_p0239.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0105_2_p0240.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0108_2_p0249.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0108_3_p0250.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0110_2_p0256.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0110_3_p0257.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0051_1_p0082.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0051_2_p0083.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0113_2_p0266.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0113_3_p0267.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0053_1_p0087.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0053_2_p0088.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0054_1_p0090.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0054_2_p0091.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0055_1_p0093.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0055_2_p0094.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0121_2_p0286.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0121_3_p0287.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0122_2_p0290.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0122_3_p0291.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0123_1_p0293.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0123_2_p0294.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0063_1_p0113.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0063_2_p0114.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0124_1_p0296.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0124_2_p0297.png',
        #-------- 100e, extra (doppler-synthesis matches)
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0003_2_p0006.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0003_2_p0006.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0005_1_p0010.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0005_1_p0010.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0005_2_p0010.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0005_2_p0010.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0055_1_p0110.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0055_1_p0110.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0060_1_p0120.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0060_1_p0120.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0065_1_p0130.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0065_1_p0130.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0068_1_p0136.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0068_1_p0136.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0068_2_p0136.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0068_2_p0136.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0072_2_p0144.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0072_2_p0144.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0073_1_p0146.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0073_1_p0146.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0063_1_p0126.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0063_1_p0126.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0074_1_p0148.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0074_1_p0148.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0075_1_p0150.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0075_1_p0150.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0079_1_p0158.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0079_1_p0158.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0082_2_p0164.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0082_2_p0164.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0084_1_p0168.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0084_1_p0168.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0086_1_p0172.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0086_1_p0172.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0088_1_p0176.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0088_1_p0176.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0088_2_p0176.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0088_2_p0176.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0089_2_p0178.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0089_2_p0178.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0093_1_p0186.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0093_1_p0186.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0097_1_p0194.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0097_1_p0194.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0102_1_p0004.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0102_1_p0004.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0107_1_p0014.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0107_1_p0014.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0109_1_p0018.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0109_1_p0018.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0110_1_p0020.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0110_1_p0020.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0115_1_p0030.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0115_1_p0030.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0119_1_p0038.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0119_1_p0038.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0126_1_p0052.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0126_1_p0052.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0129_1_p0058.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0129_1_p0058.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0130_1_p0060.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0130_1_p0060.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0134_1_p0068.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0134_1_p0068.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0136_1_p0072.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0136_1_p0072.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0138_1_p0076.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0138_1_p0076.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0141_1_p0082.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0141_1_p0082.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0142_1_p0084.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0142_1_p0084.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0144_2_p0088.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0144_2_p0088.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0147_1_p0094.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0147_1_p0094.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0148_2_p0096.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0148_2_p0096.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0150_1_p0100.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0150_1_p0100.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0151_1_p0102.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0151_1_p0102.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0153_1_p0106.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0153_1_p0106.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0154_1_p0108.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0154_1_p0108.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0154_2_p0108.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0154_2_p0108.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0156_1_p0112.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0156_1_p0112.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0158_2_p0116.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0158_2_p0116.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0160_1_p0120.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0160_1_p0120.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0161_1_p0122.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0161_1_p0122.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0165_1_p0130.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0165_1_p0130.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0166_1_p0132.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0166_1_p0132.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0167_2_p0134.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0167_2_p0134.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0004_1_p0011.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0004_1_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0008_1_p0023.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0008_1_p0023.png',
        f'{dir_mtrm_benign}/benign_nodule3_0031-0045_c0043_1_p0038.png': f'{dir_dtc_benign}/benign_nodule3_0031-0045_c0043_1_p0038.png',
        f'{dir_mtrm_benign}/benign_nodule3_0031-0045_c0045_1_p0044.png': f'{dir_dtc_benign}/benign_nodule3_0031-0045_c0045_1_p0044.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0050_1_p0011.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0050_1_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0051_2_p0013.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0051_2_p0013.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0053_1_p0017.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0053_1_p0017.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0054_1_p0019.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0054_1_p0019.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0055_1_p0021.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0055_1_p0021.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0057_1_p0025.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0057_1_p0025.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0058_1_p0027.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0058_1_p0027.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0060_1_p0031.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0060_1_p0031.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0063_1_p0037.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0063_1_p0037.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0065_1_p0041.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0065_1_p0041.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0066_2_p0043.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0066_2_p0043.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0070_1_p0051.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0070_1_p0051.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0072_1_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0072_1_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0073_1_p0007.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0073_1_p0007.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0074_1_p0009.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0074_1_p0009.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0075_1_p0011.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0075_1_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0076_1_p0013.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0076_1_p0013.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0077_1_p0015.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0077_1_p0015.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0079_1_p0019.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0079_1_p0019.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0080_1_p0021.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0080_1_p0021.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0081_1_p0023.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0081_1_p0023.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0083_1_p0027.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0083_1_p0027.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0084_1_p0029.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0084_1_p0029.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0085_1_p0031.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0085_1_p0031.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0086_1_p0033.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0086_1_p0033.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0089_2_p0039.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0089_2_p0039.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0090_1_p0041.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0090_1_p0041.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0090_2_p0041.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0090_2_p0041.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0093_1_p0047.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0093_1_p0047.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0095_1_p0051.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0095_1_p0051.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0097_1_p0055.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0097_1_p0055.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0098_2_p0057.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0098_2_p0057.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0100_2_p0061.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0100_2_p0061.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0101_1_p0003.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0101_1_p0003.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0102_1_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0102_1_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0103_2_p0007.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0103_2_p0007.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0104_2_p0009.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0104_2_p0009.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0105_2_p0011.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0105_2_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0116_1_p0033.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0116_1_p0033.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0118_2_p0037.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0118_2_p0037.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0120_2_p0041.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0120_2_p0041.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0121_1_p0043.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0121_1_p0043.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0122_1_p0045.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0122_1_p0045.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0122_2_p0045.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0122_2_p0045.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0125_1_p0051.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0125_1_p0051.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0126_2_p0053.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0126_2_p0053.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0127_1_p0055.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0127_1_p0055.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0127_2_p0055.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0127_2_p0055.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0128_1_p0057.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0128_1_p0057.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0129_2_p0059.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0129_2_p0059.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0130_1_p0061.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0130_1_p0061.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0130_2_p0061.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0130_2_p0061.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0131_2_p0063.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0131_2_p0063.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0132_1_p0065.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0132_1_p0065.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0132_2_p0065.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0132_2_p0065.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0135_1_p0071.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0135_1_p0071.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0136_1_p0073.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0136_1_p0073.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0137_1_p0075.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0137_1_p0075.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0137_2_p0075.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0137_2_p0075.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0138_2_p0077.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0138_2_p0077.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0139_1_p0079.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0139_1_p0079.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0141_1_p0083.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0141_1_p0083.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0142_2_p0085.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0142_2_p0085.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0145_2_p0091.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0145_2_p0091.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0147_2_p0095.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0147_2_p0095.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0151_1_p0003.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0151_1_p0003.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0152_1_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0152_1_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0155_1_p0011.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0155_1_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0157_1_p0015.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0157_1_p0015.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0158_1_p0017.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0158_1_p0017.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0159_1_p0019.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0159_1_p0019.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0160_1_p0021.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0160_1_p0021.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0160_2_p0021.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0160_2_p0021.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0162_1_p0025.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0162_1_p0025.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0164_2_p0029.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0164_2_p0029.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0165_1_p0031.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0165_1_p0031.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0167_1_p0035.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0167_1_p0035.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0168_1_p0037.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0168_1_p0037.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0169_1_p0039.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0169_1_p0039.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0169_2_p0039.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0169_2_p0039.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0171_1_p0043.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0171_1_p0043.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0172_1_p0045.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0172_1_p0045.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0172_3_p0045.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0172_3_p0045.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0173_1_p0047.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0173_1_p0047.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0174_1_p0049.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0174_1_p0049.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0176_1_p0053.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0176_1_p0053.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0177_1_p0055.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0177_1_p0055.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0178_1_p0057.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0178_1_p0057.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0178_2_p0057.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0178_2_p0057.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0179_1_p0059.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0179_1_p0059.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0180_1_p0061.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0180_1_p0061.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0180_2_p0061.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0180_2_p0061.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0181_1_p0063.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0181_1_p0063.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0182_1_p0065.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0182_1_p0065.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0183_3_p0067.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0183_3_p0067.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0185_1_p0071.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0185_1_p0071.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0186_2_p0073.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0186_2_p0073.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0187_1_p0075.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0187_1_p0075.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0188_1_p0077.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0188_1_p0077.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0189_1_p0079.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0189_1_p0079.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0189_2_p0079.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0189_2_p0079.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0190_1_p0081.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0190_1_p0081.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0191_3_p0083.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0191_3_p0083.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0192_1_p0085.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0192_1_p0085.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0193_1_p0087.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0193_1_p0087.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0194_1_p0089.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0194_1_p0089.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0195_1_p0091.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0195_1_p0091.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0196_1_p0093.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0196_1_p0093.png',
        f'{dir_mtrm_benign}/benign_nodule3_0151-0199_c0197_1_p0095.png': f'{dir_dtc_benign}/benign_nodule3_0151-0199_c0197_1_p0095.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0013_1_p0015.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0013_1_p0015.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0018_1_p0020.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0018_1_p0020.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0019_1_p0022.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0019_1_p0022.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0059_1_p0068.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0059_1_p0068.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0086_1_p0075.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0086_1_p0075.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0120_1_p0077.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0120_1_p0077.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0126_1_p0082.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0126_1_p0082.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0130_2_p0095.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0130_2_p0095.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0135_1_p0109.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0135_1_p0109.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0145_1_p0137.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0145_1_p0137.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0003_1_p0006.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0003_1_p0006.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0006_1_p0012.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0006_1_p0012.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0018_1_p0036.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0018_1_p0036.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0020_1_p0040.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0020_1_p0040.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0036_2_p0072.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0036_2_p0072.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0037_1_p0074.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0037_1_p0074.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0050_2_p0100.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0050_2_p0100.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0054_2_p0108.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0054_2_p0108.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0059_1_p0118.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0059_1_p0118.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0062_1_p0124.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0062_1_p0124.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0067_1_p0134.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0067_1_p0134.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0069_1_p0138.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0069_1_p0138.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0085_2_p0170.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0085_2_p0170.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0086_2_p0172.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0086_2_p0172.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0087_1_p0174.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0087_1_p0174.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0089_1_p0178.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0089_1_p0178.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0092_2_p0184.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0092_2_p0184.png',
        f'{dir_mtrm_benign}/benign_nodule1_0001-0100_c0097_2_p0194.png': f'{dir_dtc_benign}/benign_nodule1_0001-0100_c0097_2_p0194.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0118_1_p0036.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0118_1_p0036.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0132_1_p0064.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0132_1_p0064.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0152_1_p0104.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0152_1_p0104.png',
        f'{dir_mtrm_benign}/benign_nodule1_0101-0168_c0157_1_p0114.png': f'{dir_dtc_benign}/benign_nodule1_0101-0168_c0157_1_p0114.png',
        f'{dir_mtrm_benign}/benign_nodule2_0001-0016_c0005_1_p0014.png': f'{dir_dtc_benign}/benign_nodule2_0001-0016_c0005_1_p0014.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0007_2_p0020.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0007_2_p0020.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0016_1_p0047.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0016_1_p0047.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0018_2_p0053.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0018_2_p0053.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0026_1_p0077.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0026_1_p0077.png',
        f'{dir_mtrm_benign}/benign_nodule3_0001-0030_c0028_1_p0083.png': f'{dir_dtc_benign}/benign_nodule3_0001-0030_c0028_1_p0083.png',
        f'{dir_mtrm_benign}/benign_nodule3_0031-0045_c0032_1_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0031-0045_c0032_1_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule3_0031-0045_c0039_1_p0026.png': f'{dir_dtc_benign}/benign_nodule3_0031-0045_c0039_1_p0026.png',
        f'{dir_mtrm_benign}/benign_nodule3_0031-0045_c0040_2_p0029.png': f'{dir_dtc_benign}/benign_nodule3_0031-0045_c0040_2_p0029.png',
        f'{dir_mtrm_benign}/benign_nodule3_0031-0045_c0041_2_p0032.png': f'{dir_dtc_benign}/benign_nodule3_0031-0045_c0041_2_p0032.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0046_2_p0003.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0046_2_p0003.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0048_1_p0007.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0048_1_p0007.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0068_1_p0047.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0068_1_p0047.png',
        f'{dir_mtrm_benign}/benign_nodule3_0046-0070_c0069_1_p0049.png': f'{dir_dtc_benign}/benign_nodule3_0046-0070_c0069_1_p0049.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0071_1_p0003.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0071_1_p0003.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0075_2_p0011.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0075_2_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0076_2_p0013.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0076_2_p0013.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0078_1_p0017.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0078_1_p0017.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0093_2_p0047.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0093_2_p0047.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0098_1_p0057.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0098_1_p0057.png',
        f'{dir_mtrm_benign}/benign_nodule3_0071-0100_c0099_1_p0059.png': f'{dir_dtc_benign}/benign_nodule3_0071-0100_c0099_1_p0059.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0102_2_p0005.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0102_2_p0005.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0103_1_p0007.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0103_1_p0007.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0105_1_p0011.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0105_1_p0011.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0112_1_p0025.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0112_1_p0025.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0117_1_p0035.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0117_1_p0035.png',
        f'{dir_mtrm_benign}/benign_nodule3_0101-0150_c0119_1_p0039.png': f'{dir_dtc_benign}/benign_nodule3_0101-0150_c0119_1_p0039.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0020_1_p0024.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0020_1_p0024.png',
        f'{dir_mtrm_benign}/benign_siriraj_0001-0160_c0021_2_p0027.png': f'{dir_dtc_benign}/benign_siriraj_0001-0160_c0021_2_p0027.png',
        #-------- 100g, N=73, synth, Malignant-Extra(from No_Marker folder)
        f'{dir_mtrm_malignant}/malignant_nodule2_c0013_2_p0003.png': f'{dir_dtc_malignant}/malignant_nodule2_c0013_2_p0003.png',
        f'{dir_mtrm_malignant}/malignant_nodule2_c0017_3_p0006.png': f'{dir_dtc_malignant}/malignant_nodule2_c0017_3_p0006.png',
        f'{dir_mtrm_malignant}/malignant_nodule2_c0018_3_p0009.png': f'{dir_dtc_malignant}/malignant_nodule2_c0018_3_p0009.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0003_2_p0008.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0003_2_p0008.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0006_2_p0017.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0006_2_p0017.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0007_2_p0020.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0007_2_p0020.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0008_2_p0023.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0008_2_p0023.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0014_1_p0041.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0014_1_p0041.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0014_3_p0041.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0014_3_p0041.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0017_1_p0050.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0017_1_p0050.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0017_3_p0050.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0017_3_p0050.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0018_2_p0056.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0018_2_p0056.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0020_2_p0059.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0020_2_p0059.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0021_2_p0062.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0021_2_p0062.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0025_3_p0074.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0025_3_p0074.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0026_3_p0077.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0026_3_p0077.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0001-0030_c0027_2_p0080.png': f'{dir_dtc_malignant}/malignant_nodule3_0001-0030_c0027_2_p0080.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0032_4_p0005.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0032_4_p0005.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0034_2_p0011.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0034_2_p0011.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0035_1_p0014.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0035_1_p0014.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0037_2_p0020.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0037_2_p0020.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0039_1_p0026.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0039_1_p0026.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0040_1_p0029.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0040_1_p0029.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0045_2_p0044.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0045_2_p0044.png',
        f'{dir_mtrm_malignant}/malignant_nodule3_0031-0060_c0055_1_p0074.png': f'{dir_dtc_malignant}/malignant_nodule3_0031-0060_c0055_1_p0074.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0081_2_p0003.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0081_2_p0003.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0082_1_p0005.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0082_1_p0005.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0084_2_p0009.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0084_2_p0009.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0095_1_p0031.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0095_1_p0031.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0095_2_p0031.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0095_2_p0031.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0098_1_p0037.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0098_1_p0037.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0098_2_p0037.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0098_2_p0037.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0099_1_p0039.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0099_1_p0039.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0101_1_p0043.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0101_1_p0043.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0102_1_p0045.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0102_1_p0045.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0106_2_p0053.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0106_2_p0053.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0114_2_p0069.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0114_2_p0069.png',
        f'{dir_mtrm_malignant}/malignant_nodule5_0080-0123_c0117_1_p0075.png': f'{dir_dtc_malignant}/malignant_nodule5_0080-0123_c0117_1_p0075.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0009_1_p0015.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0009_1_p0015.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0010_1_p0018.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0010_1_p0018.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c001x_1_p0136.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c001x_1_p0136.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0032_1_p0035.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0032_1_p0035.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0033_1_p0038.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0033_1_p0038.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0034_1_p0041.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0034_1_p0041.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0036_1_p0044.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0036_1_p0044.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0037_1_p0047.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0037_1_p0047.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0047_1_p0071.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0047_1_p0071.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0056_1_p0096.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0056_1_p0096.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0064_1_p0116.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0064_1_p0116.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0068_1_p0122.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0068_1_p0122.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0069_1_p0126.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0069_1_p0126.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0073_1_p0140.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0073_1_p0140.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0074_1_p0144.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0074_1_p0144.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0081_1_p0165.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0081_1_p0165.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0089_1_p0186.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0089_1_p0186.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0090_1_p0190.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0090_1_p0190.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0092_1_p0197.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0092_1_p0197.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0092_2_p0198.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0092_2_p0198.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0094_1_p0203.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0094_1_p0203.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0096_1_p0210.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0096_1_p0210.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0097_1_p0214.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0097_1_p0214.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0101_1_p0226.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0101_1_p0226.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0102_1_p0230.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0102_1_p0230.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0104_1_p0237.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0104_1_p0237.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0107_1_p0244.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0107_1_p0244.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0108_1_p0248.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0108_1_p0248.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0109_1_p0252.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0109_1_p0252.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0110_1_p0255.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0110_1_p0255.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0113_1_p0265.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0113_1_p0265.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0114_1_p0269.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0114_1_p0269.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0115_1_p0273.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0115_1_p0273.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0121_1_p0285.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0121_1_p0285.png',
        f'{dir_mtrm_malignant}/malignant_siriraj_0001-0124_c0122_1_p0289.png': f'{dir_dtc_malignant}/malignant_siriraj_0001-0124_c0122_1_p0289.png',
        #-------- 100g, N=152  organic/synth, 'Siriraj Benign.zip'
        f'{dir_mtrm_benign}/Benign_38012513_C1_P2.png': f'{dir_dtc_benign}/Benign_38012513_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_38012513_C1_P3f99.png': f'{dir_dtc_benign}/Benign_38012513_C1_P4f99.png',
        f'{dir_mtrm_benign}/Benign_39001023_C1_P1.png': f'{dir_dtc_benign}/Benign_39001023_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_39001023_C1_P2f88.png': f'{dir_dtc_benign}/Benign_39001023_C1_P3f88.png',
        f'{dir_mtrm_benign}/Benign_39036847_C1_P1f77.png': f'{dir_dtc_benign}/Benign_39036847_C1_P2f77.png',
        f'{dir_mtrm_benign}/Benign_40023797_C1_P2.png': f'{dir_dtc_benign}/Benign_40023797_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_40023797_C1_P3.png': f'{dir_dtc_benign}/Benign_40023797_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_40036934_C1_P2f66.png': f'{dir_dtc_benign}/Benign_40036934_C1_P4f66.png',
        f'{dir_mtrm_benign}/Benign_40036934_C1_P3f66.png': f'{dir_dtc_benign}/Benign_40036934_C1_P4f66.png',
        f'{dir_mtrm_benign}/Benign_40055875_C1_P1f55.png': f'{dir_dtc_benign}/Benign_40055875_C1_P3f55.png',
        f'{dir_mtrm_benign}/Benign_40055875_C1_P2.png': f'{dir_dtc_benign}/Benign_40055875_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_40081562_C1_P1.png': f'{dir_dtc_benign}/Benign_40081562_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_40081562_C1_P2f44.png': f'{dir_dtc_benign}/Benign_40081562_C1_P3f44.png',
        f'{dir_mtrm_benign}/Benign_40110739_C1_P1f33.png': f'{dir_dtc_benign}/Benign_40110739_C1_P3f33.png',
        f'{dir_mtrm_benign}/Benign_40110739_C1_P2.png': f'{dir_dtc_benign}/Benign_40110739_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_40123244_C1_P1f22.png': f'{dir_dtc_benign}/Benign_40123244_C1_P4f22.png',
        f'{dir_mtrm_benign}/Benign_40123244_C1_P2f22.png': f'{dir_dtc_benign}/Benign_40123244_C1_P4f22.png',
        f'{dir_mtrm_benign}/Benign_40144364_C1_P2f11.png': f'{dir_dtc_benign}/Benign_40144364_C1_P4f11.png',
        f'{dir_mtrm_benign}/Benign_40181759_C1_P1e99.png': f'{dir_dtc_benign}/Benign_40181759_C1_P2e99.png',
        f'{dir_mtrm_benign}/Benign_41047555_C1_P1e88.png': f'{dir_dtc_benign}/Benign_41047555_C1_P3e88.png',
        f'{dir_mtrm_benign}/Benign_41047555_C1_P2.png': f'{dir_dtc_benign}/Benign_41047555_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_41138962_C1_P2e77.png': f'{dir_dtc_benign}/Benign_41138962_C1_P3e77.png',
        f'{dir_mtrm_benign}/Benign_42014757_C1_P1.png': f'{dir_dtc_benign}/Benign_42014757_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_42014757_C1_P2e66.png': f'{dir_dtc_benign}/Benign_42014757_C1_P3e66.png',
        f'{dir_mtrm_benign}/Benign_42033552_C1_P1.png': f'{dir_dtc_benign}/Benign_42033552_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_42033552_C1_P2.png': f'{dir_dtc_benign}/Benign_42033552_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_42047083_C1_P2.png': f'{dir_dtc_benign}/Benign_42047083_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_42047083_C1_P3.png': f'{dir_dtc_benign}/Benign_42047083_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_42145223_C1_P1e55.png': f'{dir_dtc_benign}/Benign_42145223_C1_P2e55.png',
        f'{dir_mtrm_benign}/Benign_42146481_C1_P1e44.png': f'{dir_dtc_benign}/Benign_42146481_C1_P2e44.png',
        f'{dir_mtrm_benign}/Benign_43047933_C1_P2.png': f'{dir_dtc_benign}/Benign_43047933_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_43063469_C1_P1e33.png': f'{dir_dtc_benign}/Benign_43063469_C1_P3e33.png',
        f'{dir_mtrm_benign}/Benign_43063469_C1_P2e33.png': f'{dir_dtc_benign}/Benign_43063469_C1_P3e33.png',
        f'{dir_mtrm_benign}/Benign_43115910_C1_P2e22.png': f'{dir_dtc_benign}/Benign_43115910_C1_P4e22.png',
        f'{dir_mtrm_benign}/Benign_43115910_C1_P3.png': f'{dir_dtc_benign}/Benign_43115910_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_43132511_C1_P1e11.png': f'{dir_dtc_benign}/Benign_43132511_C1_P2e11.png',
        f'{dir_mtrm_benign}/Benign_44059438_C1_P1d99.png': f'{dir_dtc_benign}/Benign_44059438_C1_P2d99.png',
        f'{dir_mtrm_benign}/Benign_44123445_C1_P1.png': f'{dir_dtc_benign}/Benign_44123445_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_44123445_C1_P2d88.png': f'{dir_dtc_benign}/Benign_44123445_C1_P4d88.png',
        f'{dir_mtrm_benign}/Benign_44130718_C1_P2.png': f'{dir_dtc_benign}/Benign_44130718_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_44130718_C1_P3.png': f'{dir_dtc_benign}/Benign_44130718_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_45044343_C1_P1d77.png': f'{dir_dtc_benign}/Benign_45044343_C1_P3d77.png',
        f'{dir_mtrm_benign}/Benign_45044343_C1_P2.png': f'{dir_dtc_benign}/Benign_45044343_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_46001673A_C1_P2.png': f'{dir_dtc_benign}/Benign_46001673A_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_46001673A_C1_P3.png': f'{dir_dtc_benign}/Benign_46001673A_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_46001673B_C1_P2.png': f'{dir_dtc_benign}/Benign_46001673B_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_46018725_C1_P1.png': f'{dir_dtc_benign}/Benign_46018725_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_46018725_C1_P2.png': f'{dir_dtc_benign}/Benign_46018725_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_46059897_C1_P1.png': f'{dir_dtc_benign}/Benign_46059897_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_46080236_C1_P1d55.png': f'{dir_dtc_benign}/Benign_46080236_C1_P4d55.png',
        f'{dir_mtrm_benign}/Benign_46080236_C1_P2d66.png': f'{dir_dtc_benign}/Benign_46080236_C1_P3d66.png',
        f'{dir_mtrm_benign}/Benign_47171656_C1_P1.png': f'{dir_dtc_benign}/Benign_47171656_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_47171656_C1_P2.png': f'{dir_dtc_benign}/Benign_47171656_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_47228310_C1_P1d44.png': f'{dir_dtc_benign}/Benign_47228310_C1_P4d44.png',
        f'{dir_mtrm_benign}/Benign_47228310_C1_P3d44.png': f'{dir_dtc_benign}/Benign_47228310_C1_P4d44.png',
        f'{dir_mtrm_benign}/Benign_47238762_C1_P1d33.png': f'{dir_dtc_benign}/Benign_47238762_C1_P4d33.png',
        f'{dir_mtrm_benign}/Benign_47238762_C1_P2d33.png': f'{dir_dtc_benign}/Benign_47238762_C1_P4d33.png',
        f'{dir_mtrm_benign}/Benign_47259675_C1_P1d22.png': f'{dir_dtc_benign}/Benign_47259675_C1_P2d22.png',
        f'{dir_mtrm_benign}/Benign_48110540_C1_P1.png': f'{dir_dtc_benign}/Benign_48110540_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_48121753_C1_P1d11.png': f'{dir_dtc_benign}/Benign_48121753_C1_P2d11.png',
        f'{dir_mtrm_benign}/Benign_49127749_C1_P1.png': f'{dir_dtc_benign}/Benign_49127749_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_49127749_C1_P2.png': f'{dir_dtc_benign}/Benign_49127749_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_49127749_C1_P4.png': f'{dir_dtc_benign}/Benign_49127749_C1_P4.png',
        f'{dir_mtrm_benign}/Benign_49187137_C1_P1c99.png': f'{dir_dtc_benign}/Benign_49187137_C1_P4c99.png',
        f'{dir_mtrm_benign}/Benign_49187137_C1_P2.png': f'{dir_dtc_benign}/Benign_49187137_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_49187137_C1_P3c99.png': f'{dir_dtc_benign}/Benign_49187137_C1_P4c99.png',
        f'{dir_mtrm_benign}/Benign_49211968_C1_P1.png': f'{dir_dtc_benign}/Benign_49211968_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_49211968_C1_P2.png': f'{dir_dtc_benign}/Benign_49211968_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_49214838_C1_P1c88.png': f'{dir_dtc_benign}/Benign_49214838_C1_P4c88.png',
        f'{dir_mtrm_benign}/Benign_49214838_C1_P2c88.png': f'{dir_dtc_benign}/Benign_49214838_C1_P4c88.png',
        f'{dir_mtrm_benign}/Benign_49214838_C1_P3c88.png': f'{dir_dtc_benign}/Benign_49214838_C1_P4c88.png',
        f'{dir_mtrm_benign}/Benign_50174758_C1_P1.png': f'{dir_dtc_benign}/Benign_50174758_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_50174758_C1_P2.png': f'{dir_dtc_benign}/Benign_50174758_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_50176128_C1_P1c77.png': f'{dir_dtc_benign}/Benign_50176128_C1_P2c77.png',
        f'{dir_mtrm_benign}/Benign_50268308_C1_P1.png': f'{dir_dtc_benign}/Benign_50268308_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_50268308_C1_P2.png': f'{dir_dtc_benign}/Benign_50268308_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_50268308_C1_P3.png': f'{dir_dtc_benign}/Benign_50268308_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_50314790B_C1_P2.png': f'{dir_dtc_benign}/Benign_50314790B_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_50318052_C1_P1.png': f'{dir_dtc_benign}/Benign_50318052_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_50318052_C1_P2.png': f'{dir_dtc_benign}/Benign_50318052_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_51159223B_C1_P1c66.png': f'{dir_dtc_benign}/Benign_51159223B_C1_P3c66.png',
        f'{dir_mtrm_benign}/Benign_51169278_C1_P1.png': f'{dir_dtc_benign}/Benign_51169278_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_51169278_C1_P2c55.png': f'{dir_dtc_benign}/Benign_51169278_C1_P4c55.png',
        f'{dir_mtrm_benign}/Benign_51239432_C1_P1.png': f'{dir_dtc_benign}/Benign_51239432_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_51239432_C1_P2.png': f'{dir_dtc_benign}/Benign_51239432_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_51243398_C1_P2.png': f'{dir_dtc_benign}/Benign_51243398_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_51243398_C1_P3c44.png': f'{dir_dtc_benign}/Benign_51243398_C1_P4c44.png',
        f'{dir_mtrm_benign}/Benign_52101969_C1_P1c33.png': f'{dir_dtc_benign}/Benign_52101969_C1_P3c33.png',
        f'{dir_mtrm_benign}/Benign_52101969_C1_P2.png': f'{dir_dtc_benign}/Benign_52101969_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52522026_C1_P2.png': f'{dir_dtc_benign}/Benign_52522026_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52523559_C1_P2.png': f'{dir_dtc_benign}/Benign_52523559_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52633373_C1_P1c22.png': f'{dir_dtc_benign}/Benign_52633373_C1_P3c22.png',
        f'{dir_mtrm_benign}/Benign_52633373_C1_P2.png': f'{dir_dtc_benign}/Benign_52633373_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52634272_C1_P2.png': f'{dir_dtc_benign}/Benign_52634272_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52634272_C1_P3.png': f'{dir_dtc_benign}/Benign_52634272_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_52734049_C1_P1c11.png': f'{dir_dtc_benign}/Benign_52734049_C1_P4c11.png',
        f'{dir_mtrm_benign}/Benign_52734049_C1_P2c11.png': f'{dir_dtc_benign}/Benign_52734049_C1_P4c11.png',
        f'{dir_mtrm_benign}/Benign_52734049_C1_P3c11.png': f'{dir_dtc_benign}/Benign_52734049_C1_P4c11.png',
        f'{dir_mtrm_benign}/Benign_52745655_C1_P1.png': f'{dir_dtc_benign}/Benign_52745655_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_52745655_C1_P2b99.png': f'{dir_dtc_benign}/Benign_52745655_C1_P4b99.png',
        f'{dir_mtrm_benign}/Benign_52788638_C1_P1b88.png': f'{dir_dtc_benign}/Benign_52788638_C1_P2b88.png',
        f'{dir_mtrm_benign}/Benign_52798240_C1_P2.png': f'{dir_dtc_benign}/Benign_52798240_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52798240_C1_P3.png': f'{dir_dtc_benign}/Benign_52798240_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_52920395_C1_P2.png': f'{dir_dtc_benign}/Benign_52920395_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52920395_C1_P3.png': f'{dir_dtc_benign}/Benign_52920395_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_52957622_C1_P1b77.png': f'{dir_dtc_benign}/Benign_52957622_C1_P3b77.png',
        f'{dir_mtrm_benign}/Benign_52957622_C1_P2b66.png': f'{dir_dtc_benign}/Benign_52957622_C1_P4b66.png',
        f'{dir_mtrm_benign}/Benign_52967721_C1_P1.png': f'{dir_dtc_benign}/Benign_52967721_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_52967721_C1_P2.png': f'{dir_dtc_benign}/Benign_52967721_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_52967721_C1_P3.png': f'{dir_dtc_benign}/Benign_52967721_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_53018304_C1_P1.png': f'{dir_dtc_benign}/Benign_53018304_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_53018304_C1_P2.png': f'{dir_dtc_benign}/Benign_53018304_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53018304_C1_P3.png': f'{dir_dtc_benign}/Benign_53018304_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_53290031_C1_P1b55.png': f'{dir_dtc_benign}/Benign_53290031_C1_P4b55.png',
        f'{dir_mtrm_benign}/Benign_53290031_C1_P3.png': f'{dir_dtc_benign}/Benign_53290031_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_53323484_C1_P1.png': f'{dir_dtc_benign}/Benign_53323484_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_53323484_C1_P2.png': f'{dir_dtc_benign}/Benign_53323484_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53335810A_C1_P1.png': f'{dir_dtc_benign}/Benign_53335810A_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_53335810A_C1_P2b44.png': f'{dir_dtc_benign}/Benign_53335810A_C1_P3b44.png',
        f'{dir_mtrm_benign}/Benign_53335810B_C1_P2b33.png': f'{dir_dtc_benign}/Benign_53335810B_C1_P4b33.png',
        f'{dir_mtrm_benign}/Benign_53405237_C1_P1.png': f'{dir_dtc_benign}/Benign_53405237_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_53405237_C1_P2.png': f'{dir_dtc_benign}/Benign_53405237_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53405237_C1_P4.png': f'{dir_dtc_benign}/Benign_53405237_C1_P4.png',
        f'{dir_mtrm_benign}/Benign_53431221_C1_P1b22.png': f'{dir_dtc_benign}/Benign_53431221_C1_P3b22.png',
        f'{dir_mtrm_benign}/Benign_53431782_C1_P1b11.png': f'{dir_dtc_benign}/Benign_53431782_C1_P3b11.png',
        f'{dir_mtrm_benign}/Benign_53431782_C1_P2.png': f'{dir_dtc_benign}/Benign_53431782_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53444043_C1_P2.png': f'{dir_dtc_benign}/Benign_53444043_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53566041_C1_P1.png': f'{dir_dtc_benign}/Benign_53566041_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_53610546_C1_P2.png': f'{dir_dtc_benign}/Benign_53610546_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53643924_C1_P1a99.png': f'{dir_dtc_benign}/Benign_53643924_C1_P4a99.png',
        f'{dir_mtrm_benign}/Benign_53643924_C1_P2.png': f'{dir_dtc_benign}/Benign_53643924_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53779961_C1_P2a88.png': f'{dir_dtc_benign}/Benign_53779961_C1_P4a88.png',
        f'{dir_mtrm_benign}/Benign_53779961_C1_P3.png': f'{dir_dtc_benign}/Benign_53779961_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_53847621_C1_P1.png': f'{dir_dtc_benign}/Benign_53847621_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_53942796A_C1_P1a77.png': f'{dir_dtc_benign}/Benign_53942796A_C1_P3a77.png',
        f'{dir_mtrm_benign}/Benign_53942796A_C1_P2.png': f'{dir_dtc_benign}/Benign_53942796A_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_53942796B_C1_P1a66.png': f'{dir_dtc_benign}/Benign_53942796B_C1_P3a66.png',
        f'{dir_mtrm_benign}/Benign_53942796B_C1_P2.png': f'{dir_dtc_benign}/Benign_53942796B_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_54052992_C1_P2a55.png': f'{dir_dtc_benign}/Benign_54052992_C1_P4a55.png',
        f'{dir_mtrm_benign}/Benign_54052992_C1_P3.png': f'{dir_dtc_benign}/Benign_54052992_C1_P3.png',
        f'{dir_mtrm_benign}/Benign_54235576_C1_P1.png': f'{dir_dtc_benign}/Benign_54235576_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_54235576_C1_P2.png': f'{dir_dtc_benign}/Benign_54235576_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_54300962_C1_P3a44.png': f'{dir_dtc_benign}/Benign_54300962_C1_P4a44.png',
        f'{dir_mtrm_benign}/Benign_54370507_C1_P1a33.png': f'{dir_dtc_benign}/Benign_54370507_C1_P3a33.png',
        f'{dir_mtrm_benign}/Benign_54370507_C1_P2.png': f'{dir_dtc_benign}/Benign_54370507_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_54371413_C1_P1a22.png': f'{dir_dtc_benign}/Benign_54371413_C1_P2a22.png',
        f'{dir_mtrm_benign}/Benign_54372643_C1_P1.png': f'{dir_dtc_benign}/Benign_54372643_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_54372643_C1_P2a11.png': f'{dir_dtc_benign}/Benign_54372643_C1_P3a11.png',
        f'{dir_mtrm_benign}/Benign_54389827_C1_P1.png': f'{dir_dtc_benign}/Benign_54389827_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_54389827_C1_P2.png': f'{dir_dtc_benign}/Benign_54389827_C1_P2.png',
        f'{dir_mtrm_benign}/Benign_54391997A_C1_P1.png': f'{dir_dtc_benign}/Benign_54391997A_C1_P1.png',
        f'{dir_mtrm_benign}/Benign_54391997A_C1_P2.png': f'{dir_dtc_benign}/Benign_54391997A_C1_P2.png',
        #-------- 100g, N=65   organic/synth, 'Siriraj Malignat.zip'
        f'{dir_mtrm_malignant}/Malignant_39035974_C1_P2b33.png': f'{dir_dtc_malignant}/Malignant_39035974_C1_P3b33.png',
        f'{dir_mtrm_malignant}/Malignant_39053545A_C1_P1.png': f'{dir_dtc_malignant}/Malignant_39053545A_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_39053545A_C1_P2.png': f'{dir_dtc_malignant}/Malignant_39053545A_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_39053545B_C1_P1.png': f'{dir_dtc_malignant}/Malignant_39053545B_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_39151279_C1_P1b22.png': f'{dir_dtc_malignant}/Malignant_39151279_C1_P3b22.png',
        f'{dir_mtrm_malignant}/Malignant_39151279_C1_P2b22.png': f'{dir_dtc_malignant}/Malignant_39151279_C1_P3b22.png',
        f'{dir_mtrm_malignant}/Malignant_41175270_C1_P1.png': f'{dir_dtc_malignant}/Malignant_41175270_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_41175270_C1_P2b11.png': f'{dir_dtc_malignant}/Malignant_41175270_C1_P3b11.png',
        f'{dir_mtrm_malignant}/Malignant_42007064_C1_P2.png': f'{dir_dtc_malignant}/Malignant_42007064_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_42043029_C1_P1a99.png': f'{dir_dtc_malignant}/Malignant_42043029_C1_P3a99.png',
        f'{dir_mtrm_malignant}/Malignant_42043029_C1_P2.png': f'{dir_dtc_malignant}/Malignant_42043029_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_42176690_C1_P2.png': f'{dir_dtc_malignant}/Malignant_42176690_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_42184362_C1_P2a88.png': f'{dir_dtc_malignant}/Malignant_42184362_C1_P4a88.png',
        f'{dir_mtrm_malignant}/Malignant_42184362_C1_P3.png': f'{dir_dtc_malignant}/Malignant_42184362_C1_P3.png',
        f'{dir_mtrm_malignant}/Malignant_43003731_C1_P1.png': f'{dir_dtc_malignant}/Malignant_43003731_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_43003731_C1_P2a77.png': f'{dir_dtc_malignant}/Malignant_43003731_C1_P3a77.png',
        f'{dir_mtrm_malignant}/Malignant_43045284_C1_P3a66.png': f'{dir_dtc_malignant}/Malignant_43045284_C1_P4a66.png',
        f'{dir_mtrm_malignant}/Malignant_43134277_C1_P2.png': f'{dir_dtc_malignant}/Malignant_43134277_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_44105269_C1_P1.png': f'{dir_dtc_malignant}/Malignant_44105269_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_44105269_C1_P2.png': f'{dir_dtc_malignant}/Malignant_44105269_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_45061183_C1_P3a55.png': f'{dir_dtc_malignant}/Malignant_45061183_C1_P4a55.png',
        f'{dir_mtrm_malignant}/Malignant_46159152_C1_P1a44.png': f'{dir_dtc_malignant}/Malignant_46159152_C1_P3a44.png',
        f'{dir_mtrm_malignant}/Malignant_48207477_C1_P1a33.png': f'{dir_dtc_malignant}/Malignant_48207477_C1_P3a33.png',
        f'{dir_mtrm_malignant}/Malignant_48207477_C1_P2.png': f'{dir_dtc_malignant}/Malignant_48207477_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_50203629_C1_P1.png': f'{dir_dtc_malignant}/Malignant_50203629_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_50314790A_C1_P1a22.png': f'{dir_dtc_malignant}/Malignant_50314790A_C1_P3a22.png',
        f'{dir_mtrm_malignant}/Malignant_50317418_C1_P2.png': f'{dir_dtc_malignant}/Malignant_50317418_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_52631103A_C1_P1.png': f'{dir_dtc_malignant}/Malignant_52631103A_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_52669318_C1_P1a11.png': f'{dir_dtc_malignant}/Malignant_52669318_C1_P2a11.png',
        f'{dir_mtrm_malignant}/Malignant_52727665_C1_P1x99.png': f'{dir_dtc_malignant}/Malignant_52727665_C1_P2x99.png',
        f'{dir_mtrm_malignant}/Malignant_52830743A_C1_P2x88.png': f'{dir_dtc_malignant}/Malignant_52830743A_C1_P4x88.png',
        f'{dir_mtrm_malignant}/Malignant_52830743A_C1_P3.png': f'{dir_dtc_malignant}/Malignant_52830743A_C1_P3.png',
        f'{dir_mtrm_malignant}/Malignant_53038280_C1_P1x77.png': f'{dir_dtc_malignant}/Malignant_53038280_C1_P3x77.png',
        f'{dir_mtrm_malignant}/Malignant_53038280_C1_P2.png': f'{dir_dtc_malignant}/Malignant_53038280_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_53152975_C1_P1.png': f'{dir_dtc_malignant}/Malignant_53152975_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_53152975_C1_P3.png': f'{dir_dtc_malignant}/Malignant_53152975_C1_P3.png',
        f'{dir_mtrm_malignant}/Malignant_53191134_C1_P2x66.png': f'{dir_dtc_malignant}/Malignant_53191134_C1_P4x66.png',
        f'{dir_mtrm_malignant}/Malignant_53191134_C1_P3.png': f'{dir_dtc_malignant}/Malignant_53191134_C1_P3.png',
        f'{dir_mtrm_malignant}/Malignant_53254085_C1_P1.png': f'{dir_dtc_malignant}/Malignant_53254085_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_53254085_C1_P2x55.png': f'{dir_dtc_malignant}/Malignant_53254085_C1_P3x55.png',
        f'{dir_mtrm_malignant}/Malignant_53389678A_C1_P2.png': f'{dir_dtc_malignant}/Malignant_53389678A_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_53501904_C1_P1x44.png': f'{dir_dtc_malignant}/Malignant_53501904_C1_P3x44.png',
        f'{dir_mtrm_malignant}/Malignant_53501904_C1_P2.png': f'{dir_dtc_malignant}/Malignant_53501904_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_53542205_C1_P1.png': f'{dir_dtc_malignant}/Malignant_53542205_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_53542205_C1_P2.png': f'{dir_dtc_malignant}/Malignant_53542205_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_53635697A_C1_P2x33.png': f'{dir_dtc_malignant}/Malignant_53635697A_C1_P3x33.png',
        f'{dir_mtrm_malignant}/Malignant_53639494A_C1_P1.png': f'{dir_dtc_malignant}/Malignant_53639494A_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_53639494A_C1_P3x22.png': f'{dir_dtc_malignant}/Malignant_53639494A_C1_P4x22.png',
        f'{dir_mtrm_malignant}/Malignant_53836084_C1_P1x11.png': f'{dir_dtc_malignant}/Malignant_53836084_C1_P2x11.png',
        f'{dir_mtrm_malignant}/Malignant_53912007_C1_P1ii.png': f'{dir_dtc_malignant}/Malignant_53912007_C1_P2ii.png',
        f'{dir_mtrm_malignant}/Malignant_53971667_C1_P2.png': f'{dir_dtc_malignant}/Malignant_53971667_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_53971667_C1_P3hh.png': f'{dir_dtc_malignant}/Malignant_53971667_C1_P4hh.png',
        f'{dir_mtrm_malignant}/Malignant_54082584_C1_P1gg.png': f'{dir_dtc_malignant}/Malignant_54082584_C1_P3gg.png',
        f'{dir_mtrm_malignant}/Malignant_54082584_C1_P2.png': f'{dir_dtc_malignant}/Malignant_54082584_C1_P2.png',
        f'{dir_mtrm_malignant}/Malignant_54292429_C1_P1.png': f'{dir_dtc_malignant}/Malignant_54292429_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_54395362_C1_P1.png': f'{dir_dtc_malignant}/Malignant_54395362_C1_P1.png',
        f'{dir_mtrm_malignant}/Malignant_54395362_C1_P2ff.png': f'{dir_dtc_malignant}/Malignant_54395362_C1_P3ff.png',
        f'{dir_mtrm_malignant}/Malignant_54396864_C1_P1ee.png': f'{dir_dtc_malignant}/Malignant_54396864_C1_P4ee.png',
        f'{dir_mtrm_malignant}/Malignant_54401487_C1_P2dd.png': f'{dir_dtc_malignant}/Malignant_54401487_C1_P4dd.png',
        f'{dir_mtrm_malignant}/Malignant_54401487_C1_P3.png': f'{dir_dtc_malignant}/Malignant_54401487_C1_P3.png',
        f'{dir_mtrm_malignant}/Malignant_54403879_C1_P3.png': f'{dir_dtc_malignant}/Malignant_54403879_C1_P3.png',
        f'{dir_mtrm_malignant}/Malignant_54409222_C1_P1cc.png': f'{dir_dtc_malignant}/Malignant_54409222_C1_P2cc.png',
        f'{dir_mtrm_malignant}/Malignant_54413799A_C1_P1bb.png': f'{dir_dtc_malignant}/Malignant_54413799A_C1_P2bb.png',
        f'{dir_mtrm_malignant}/Malignant_54413801_C1_P1aa.png': f'{dir_dtc_malignant}/Malignant_54413801_C1_P3aa.png',
        f'{dir_mtrm_malignant}/Malignant_54413801_C1_P2aa.png': f'{dir_dtc_malignant}/Malignant_54413801_C1_P3aa.png',
        #-------- 100g, N=55, Benign_Remove, synth < 'Dataset_doppler_100g_Synthesized.zip'
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0006_2_p0012.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0006_2_p0012.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0009_2_p0018.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0009_2_p0018.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0013_1_p0026.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0013_1_p0026.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0014_1_p0028.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0014_1_p0028.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0016_1_p0032.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0016_1_p0032.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0017_1_p0034.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0017_1_p0034.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0017_2_p0034.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0017_2_p0034.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0018_2_p0036.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0018_2_p0036.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0019_1_p0038.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0019_1_p0038.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0021_2_p0042.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0021_2_p0042.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0023_1_p0046.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0023_1_p0046.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0023_2_p0046.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0023_2_p0046.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0026_1_p0052.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0026_1_p0052.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0027_1_p0054.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0027_1_p0054.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0028_1_p0056.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0028_1_p0056.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0030_1_p0060.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0030_1_p0060.png',
        f'{dir_mtrm_validate_benign}/benign_nodule1_0001-0100_c0044_1_p0088.png': f'{dir_dtc_validate_benign}/benign_nodule1_0001-0100_c0044_1_p0088.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0011_2_p0032.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0011_2_p0032.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0015_1_p0044.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0015_1_p0044.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0015_2_p0044.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0015_2_p0044.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0017_1_p0050.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0017_1_p0050.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0018_1_p0053.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0018_1_p0053.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0020_1_p0059.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0020_1_p0059.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0021_1_p0062.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0021_1_p0062.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0022_1_p0065.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0022_1_p0065.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0023_1_p0068.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0023_1_p0068.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0027_1_p0080.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0027_1_p0080.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0028_2_p0083.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0028_2_p0083.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0029_2_p0086.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0029_2_p0086.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0001-0030_c0030_2_p0089.png': f'{dir_dtc_validate_benign}/benign_nodule3_0001-0030_c0030_2_p0089.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0031-0045_c0031_1_p0002.png': f'{dir_dtc_validate_benign}/benign_nodule3_0031-0045_c0031_1_p0002.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0031-0045_c0033_1_p0008.png': f'{dir_dtc_validate_benign}/benign_nodule3_0031-0045_c0033_1_p0008.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0031-0045_c0036_1_p0017.png': f'{dir_dtc_validate_benign}/benign_nodule3_0031-0045_c0036_1_p0017.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0031-0045_c0037_1_p0020.png': f'{dir_dtc_validate_benign}/benign_nodule3_0031-0045_c0037_1_p0020.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0031-0045_c0038_1_p0023.png': f'{dir_dtc_validate_benign}/benign_nodule3_0031-0045_c0038_1_p0023.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0128_2_p0057.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0128_2_p0057.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0131_1_p0063.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0131_1_p0063.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0133_2_p0067.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0133_2_p0067.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0138_1_p0077.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0138_1_p0077.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0144_1_p0089.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0144_1_p0089.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0146_1_p0093.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0146_1_p0093.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0101-0150_c0148_2_p0097.png': f'{dir_dtc_validate_benign}/benign_nodule3_0101-0150_c0148_2_p0097.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0154_1_p0009.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0154_1_p0009.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0162_2_p0025.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0162_2_p0025.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0164_1_p0029_1.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0164_1_p0029_1.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0164_1_p0029_2.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0164_1_p0029_2.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0166_1_p0033.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0166_1_p0033.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0167_2_p0035.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0167_2_p0035.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0183_1_p0067.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0183_1_p0067.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0186_1_p0073.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0186_1_p0073.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0187_3_p0075.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0187_3_p0075.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0191_1_p0083.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0191_1_p0083.png',
        f'{dir_mtrm_validate_benign}/benign_nodule3_0151-0199_c0199_1_p0099.png': f'{dir_dtc_validate_benign}/benign_nodule3_0151-0199_c0199_1_p0099.png',
        f'{dir_mtrm_validate_benign}/benign_siriraj_0001-0160_c0065_1_p0070.png': f'{dir_dtc_validate_benign}/benign_siriraj_0001-0160_c0065_1_p0070.png',
        f'{dir_mtrm_validate_benign}/benign_siriraj_0001-0160_c0144_1_p0135.png': f'{dir_dtc_validate_benign}/benign_siriraj_0001-0160_c0144_1_p0135.png',
        #-------- 100g, N=55, Malignant_Remove, synth < 'Dataset_doppler_100g_Synthesized.zip'
        f'{dir_mtrm_validate_malignant}/malignant_nodule2_c0013_4_p0003.png': f'{dir_dtc_validate_malignant}/malignant_nodule2_c0013_4_p0003.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule2_c0017_2_p0006.png': f'{dir_dtc_validate_malignant}/malignant_nodule2_c0017_2_p0006.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0009_1_p0026.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0009_1_p0026.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0009_2_p0026.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0009_2_p0026.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0010_1_p0029.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0010_1_p0029.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0011_1_p0032.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0011_1_p0032.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0015_1_p0044.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0015_1_p0044.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0016_1_p0047.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0016_1_p0047.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0018_1_p0056.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0018_1_p0056.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0018_3_p0056.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0018_3_p0056.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0022_1_p0065.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0022_1_p0065.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0023_1_p0068.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0023_1_p0068.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0023_2_p0068.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0023_2_p0068.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0026_1_p0077.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0026_1_p0077.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0029_2_p0086.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0029_2_p0086.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0030_1_p0089.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0030_1_p0089.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0001-0030_c0030_2_p0089.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0001-0030_c0030_2_p0089.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0031_1_p0002.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0031_1_p0002.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0031_2_p0002.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0031_2_p0002.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0032_1_p0005.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0032_1_p0005.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0032_2_p0005.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0032_2_p0005.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0036_1_p0017.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0036_1_p0017.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0037_1_p0020.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0037_1_p0020.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0038_2_p0023.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0038_2_p0023.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0039_2_p0026.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0039_2_p0026.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0042_1_p0035.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0042_1_p0035.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0042_2_p0035.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0042_2_p0035.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0043_1_p0038.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0043_1_p0038.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0043_2_p0038.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0043_2_p0038.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0044_1_p0041.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0044_1_p0041.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0058_1_p0083.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0058_1_p0083.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0058_2_p0083.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0058_2_p0083.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0058_3_p0083.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0058_3_p0083.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0059_1_p0086.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0059_1_p0086.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0059_2_p0086.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0059_2_p0086.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule3_0031-0060_c0060_1_p0089.png': f'{dir_dtc_validate_malignant}/malignant_nodule3_0031-0060_c0060_1_p0089.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule5_0080-0123_c0083_1_p0007.png': f'{dir_dtc_validate_malignant}/malignant_nodule5_0080-0123_c0083_1_p0007.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule5_0080-0123_c0091_1_p0023.png': f'{dir_dtc_validate_malignant}/malignant_nodule5_0080-0123_c0091_1_p0023.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule5_0080-0123_c0099_2_p0039.png': f'{dir_dtc_validate_malignant}/malignant_nodule5_0080-0123_c0099_2_p0039.png',
        f'{dir_mtrm_validate_malignant}/malignant_nodule5_0080-0123_c0104_1_p0049.png': f'{dir_dtc_validate_malignant}/malignant_nodule5_0080-0123_c0104_1_p0049.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0011_2_p0019.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0011_2_p0019.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0012_1_p0025.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0012_1_p0025.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0014_1_p0027.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0014_1_p0027.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0033_2_p0039.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0033_2_p0039.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0036_2_p0045.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0036_2_p0045.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0037_2_p0048.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0037_2_p0048.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0056_2_p0097.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0056_2_p0097.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0062_1_p0111.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0062_1_p0111.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0067_1_p0120.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0067_1_p0120.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0074_2_p0145.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0074_2_p0145.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0083_1_p0172.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0083_1_p0172.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0109_2_p0253.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0109_2_p0253.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0116_1_p0277.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0116_1_p0277.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0117_1_p0279.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0117_1_p0279.png',
        f'{dir_mtrm_validate_malignant}/malignant_siriraj_0001-0124_c0118_1_p0281.png': f'{dir_dtc_validate_malignant}/malignant_siriraj_0001-0124_c0118_1_p0281.png',
        #--------
    }
    return to_doppler

#

def bbox_to_hw_slices(bbox):
    return (
        slice(int(bbox[1]), int(bbox[3])),  # i.e. height_min:height_max
        slice(int(bbox[0]), int(bbox[2])))  # i.e. width_min:width_max

def bbox_draw(img, bbox, color=(255, 0, 0), thickness=1):
    return cv2.rectangle(img,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])), color, thickness)


def get_path_doppler(train_img_path):
    #print('@@ train_img_path:', train_img_path)
    dataset_doppler_root = train_img_path.split('/')[0]
    to_doppler = get_to_doppler(dataset_doppler_root)
    path_doppler = to_doppler[train_img_path] if train_img_path in to_doppler else None
    #print('@@ path_doppler:', path_doppler)

    return path_doppler

def get_bbox_doppler(path_doppler, size):
    # get doppler bbox (scaled)
    raw = cv2.imread(path_doppler)
    if raw is None:
        raise ValueError(f'invalid `raw` for: {path_doppler}')
    bbox_raw = detect_doppler(raw)
    if bbox_raw is None:
        logger.debug(f'detect_doppler() failed for: {path_doppler}; using `bbox_crop` instead')
        return None

    bbox = np.array([
        bbox_raw[0] * size[0] / raw.shape[1], bbox_raw[1] * size[1] / raw.shape[0],
        bbox_raw[2] * size[0] / raw.shape[1], bbox_raw[3] * size[1] / raw.shape[0]],
        dtype=np.float32)

    if bbox[2] - bbox[0] < 1. or bbox[3] - bbox[1] < 1.:
        logger.debug('doppler `bbox` too squeezed due to scaling; using `bbox_crop` instead')
        return None

    return bbox

def resolve_hw_slices(bbox_crop, train_img_copy, train_img_path, idx, size, savepath, config):

    path_doppler = get_path_doppler(train_img_path)
    if 1 and path_doppler is None:  # strict check
        raise ValueError(f'`path_doppler` not found for: {train_img_path}')

    if path_doppler:
        bbox = get_bbox_doppler(path_doppler, size)
        if bbox is None:
            return bbox_to_hw_slices(bbox_crop)

        iou, isec_in_crop = get_iou(bbox, bbox_crop)

        force_doppler = config.get('thresh_force_doppler_in_crop')
        if force_doppler is not None and force_doppler:
            qualify = 0
        else:
            thresh_isec_in_crop = config['thresh_isec_in_crop']  # the key must exist
            logger.debug(f'thresh_isec_in_crop: {thresh_isec_in_crop}')
            qualify = 1 if iou > 1e-4 and isec_in_crop > thresh_isec_in_crop else 0

        digest = hashlib.md5(path_doppler.encode('utf-8')).hexdigest()
        debug_fname_jpg = f'debug_crop_doppler_{idx}_iou_%0.4f_isecincrop_%0.3f_qualify_%d_digest_%s.jpg' % (
            iou, isec_in_crop, qualify, digest)
        logger.debug(f'debug_fname_jpg: {debug_fname_jpg}')

        if savepath is not None:  # debug dump
            bbox_draw(train_img_copy, bbox, (255, 255, 0), 1)  # blue
            bbox_draw(train_img_copy, bbox_crop, (0, 0, 255), 1)  # red
            cv2.imwrite(os.path.join(savepath, debug_fname_jpg), train_img_copy)

            # crop patch image; OK
            sh_, sw_ = bbox_to_hw_slices(bbox_crop)
            img_ = train_img_copy.copy()[sh_, sw_, :]
            cv2.imwrite(os.path.join(savepath, f'debug_crop_idx_{idx}.jpg'), img_)

            # doppler patch image; OK
            sh_, sw_ = bbox_to_hw_slices(bbox)
            img_ = train_img_copy.copy()[sh_, sw_, :]
            cv2.imwrite(os.path.join(savepath, f'debug_doppler_idx_{idx}.jpg'), img_)

    return bbox_to_hw_slices(bbox_crop if qualify else bbox)
