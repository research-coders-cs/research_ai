from .demo import test as demo_test
from .demo import train as demo_train
from .demo import train_with_doppler as demo_train_with_doppler
from .shim import stat_ds_paths, random_split_ds_path, build_dataset, \
    get_mnist_ds_paths, get_thyroid_ds_paths, get_mri_ds_paths

import logging
logger = logging.getLogger('@@')
logger.setLevel(level=logging.DEBUG if 1 else logging.INFO)


def main():

    CLASS_NAMES_SORTED_THYROID = ['benign', 'malignant']

    if 0:  # adaptation of 'compare.{ipynb,py}' exported from https://colab.research.google.com/drive/1kxMFgo1LyVqPYqhS6_UJKUsVvA2-l9wk
        from .demo import doppler_compare
        doppler_compare()

    if 0:
        # ckpt = 'ttt/51/output/demo_train/densenet_250_8_lr-1e5_n4_75.000'  # 0.800
        # demo.test(ckpt)  # TODO - generate 'confusion_matrix_test-*.png', 'test-*.png'

        from .demo import TEST_DS_PATH_DEFAULT
        ckpt = 'densenet_224_8_lr-1e5_n4_95.968.ckpt'  # 0.9xx, LGTM
        demo_test(ckpt, CLASS_NAMES_SORTED_THYROID, 'densenet121', TEST_DS_PATH_DEFAULT, 224, 8)

    if 0:
        total_epochs = 2
        #model = 'densenet121'
        model = 'resnet34'

        #dataset_doppler = 'Dataset_doppler_100e'
        dataset_doppler = 'Dataset_doppler_100g'

        ds_paths = {
            'train': build_dataset({
                'benign': ['Markers_Train_Remove_Markers/Benign_Remove/train'],
                'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/train'],
            }, root=dataset_doppler),  # 70% + extra, 70% (doppler matched)
            'validate': build_dataset({
                'benign': ['Markers_Train_Remove_Markers/Benign_Remove/validate'],
                'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/validate'],
            }, root=dataset_doppler),  # 30% 30% (doppler matched)
        }

        stat_ds_paths(ds_paths)

        #ckpt = demo_train(total_epochs, model, ds_paths)
        ckpt = demo_train_with_doppler(total_epochs, model, ds_paths, {  # <-- config_doppler=None
            'thresh_isec_in_crop': 0.25,
            #'thresh_isec_in_crop': 0.50,
            #'thresh_isec_in_crop': 0.75,
            #'thresh_force_doppler_in_crop': True,
            'disable_doppler_crop': False,
            'disable_doppler_drop': False,
        })

        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root=dataset_doppler)  # 75 75

        demo_test(ckpt, CLASS_NAMES_SORTED_THYROID, model, test_ds_path)

    if 0:  # experiment - 'heatmap-compare-doppler_100c-TrueFalse--rounds--tests_100d'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/output/demo_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/run-2/output/demo_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-False/run-3/output/demo_train/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/output/demo_train_with_doppler/resnet34_250_8_lr-1e5_n4'
        ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/run-2/output/demo_train_with_doppler/resnet34_250_8_lr-1e5_n4'
        # ckpt = './heatmap-compare-doppler_100c-TrueFalse--rounds/train-test-doppler-True/run-3/output/demo_train_with_doppler/resnet34_250_8_lr-1e5_n4'

        demo_test(ckpt, CLASS_NAMES_SORTED_THYROID, 'resnet34', build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d'))

    if 0:  # demo - acc 0.65-0.68
        ckpt = 'WSDAN_doppler_100d-resnet34_250_8_lr-1e5_n4.ckpt'
        test_ds_path = build_dataset({
            'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
            'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
        }, root='Dataset_doppler_100d')

        demo_test(ckpt, CLASS_NAMES_SORTED_THYROID, 'resnet34', test_ds_path, 250, 8)

    if 0:  # experiment - default
        total_epochs = 10
        model = 'resnet34'

        ckpt = demo_train(total_epochs, model)
        demo_test(ckpt, CLASS_NAMES_SORTED_THYROID, model)

    if 0:  # !!!! k-fold dev
        #total_epochs = 1
        total_epochs = 2
        model = 'resnet34'

        kfold = build_dataset({
            'benign': ['Train/Benign', 'Val/Benign'],
            'malignant': ['Train/Malignant', 'Val/Malignant'],
        }, root='Dataset_train_test_val')
        kfold['benign'] = kfold['benign'][0:30]
        kfold['malignant'] = kfold['malignant'][0:25]

        ds_paths = {
            'kfold': kfold,  # 30 25 --(to_be_truncated)--> 30 24
            'kfold_slices_val': [  # k=3
                [slice(0, 10), slice(0, 8)],
                [slice(10, 20), slice(8, 16)],
                [slice(20, 30), slice(16, 24)],
            ],
        }
        ckpt = demo_train(total_epochs, model, ds_paths)
        demo_test(ckpt, CLASS_NAMES_SORTED_THYROID, model)

    if 0:  # siriraj_original_Testset_26.zip
        test_ds_path = build_dataset({
            'benign': ['test26/Benign'],
            'malignant': ['test26/Malignant'],
        }, root='siriraj_original_Testset_26')

        ##demo_test('xxxx/ckpt', CLASS_NAMES_SORTED_THYROID, 'resnet34', test_ds_path)
        demo_test('densenet121_250_8_lr-1e5_n4', CLASS_NAMES_SORTED_THYROID, 'densenet121', test_ds_path)

    if 1:  # non-binary classification
        total_epochs = 2
        model = 'resnet34'

        #====
        ds_paths, class_names_sorted = get_thyroid_ds_paths('100g', debug=False)
        print(class_names_sorted)
        #ckpt = demo_train(total_epochs, model, ds_paths)
        #====
        if 0:
            ds_paths, class_names_sorted = get_mnist_ds_paths(  # mnist 10-class
                root_train='datasets_vit/pngs/train--sparse',  # 128 samples
                root_test='datasets_vit/pngs/test--sparse')  # 10 samples

            stat_ds_paths(ds_paths)

            ds_path_train, ds_path_validate = random_split_ds_path(ds_paths['train'], [110, 18])
            ds_path_test = ds_paths['test']

            ds_paths_adapted = { 'train': ds_path_train, 'validate': ds_path_validate }
            stat_ds_paths(ds_paths_adapted)

            if 0:  # !!
                ckpt = demo_train(total_epochs, model, ds_paths_adapted)

            #ckpt = 'output--pc-wsdan-mnist-sparse/demo_train/resnet34_250_8_lr-1e5_n4'  # 2/10 poor
            #ckpt = 'output--colab-wsdan-mnist/eps2/resnet34_250_8_lr-1e5_n4'  # eps=2, test 4/10 poor
            ckpt = 'output--colab-wsdan-mnist/eps8/resnet34_250_8_lr-1e5_n4'  # eps=8, test 7/10

            demo_test(ckpt, class_names_sorted, model, ds_path_test)
        #====
        if 1:
            ds_paths, class_names_sorted = get_mri_ds_paths(
                #'erica', root='datasets_mri/50-001')  # colab --> 1200
                'erica', root='datasets_mri/50-001-100')  # debug --> 202

            stat_ds_paths(ds_paths)

            ds_path_train, ds_path_validate, ds_path_test = random_split_ds_path(
                #ds_paths['train'], [1000, 100, 100])  # colab <-- 1200
                ds_paths['train'], [140, 20, 42])  # debug <-- 202

            ds_paths_adapted = { 'train': ds_path_train, 'validate': ds_path_validate }
            stat_ds_paths(ds_paths_adapted)

            if 0:  # !!
                ckpt = demo_train(total_epochs, model, ds_paths_adapted, mri_ch=250, mri_rh=80)
            else:
                ckpt = 'wsdan_erica_colab_20eps--resnet34_250_8_lr-1e5_n4.pth'

            demo_test(ckpt, class_names_sorted, model, ds_path_test, mri_ch=250, mri_rh=80)
        #====


if __name__ == '__main__':
    main()
