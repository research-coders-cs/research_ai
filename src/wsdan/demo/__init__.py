import torch
from torch.utils.data import DataLoader

from ..digitake.preprocess import build_dataset

from ..net import WSDAN, net_train, net_test

from .transform import ThyroidDataset, get_transform##, get_transform_center_crop, transform_fn
from .utils import mk_artifact_dir, get_device


WSDAN_NUM_CLASSES = 2

TRAIN_DS_PATH_DEFAULT = build_dataset({
    'benign': ['Train/Benign'],
    'malignant': ['Train/Malignant'],
}, root='Dataset_train_test_val')  # 21 20

VALIDATE_DS_PATH_DEFAULT = build_dataset({
    'benign': ['Val/Benign'],
    'malignant': ['Val/Malignant'],
}, root='Dataset_train_test_val')  # 10 10

TEST_DS_PATH_DEFAULT = build_dataset({
    'benign': ['Test/Benign'],
    'malignant': ['Test/Malignant'],
}, root='Dataset_train_test_val')  # 10 10

TOTAL_EPOCHS_DEFAULT = 100
MODEL_DEFAULT = 'densenet121'

print("@@ torch.__version__:", torch.__version__)


def doppler_compare():
    import os
    from ..net.doppler import doppler_comp, get_iou, plot_comp, get_sample_paths
    import matplotlib.pyplot as plt
    savepath = mk_artifact_dir('demo_doppler_comp')

    for path_doppler, path_markers, path_markers_label in get_sample_paths():
        print('\n@@ -------- calling doppler_comp() for')
        print(f'  {os.path.basename(path_doppler)} vs')
        print(f'  {os.path.basename(path_markers)}')

        bbox_doppler, bbox_markers, border_img_doppler, border_img_markers = doppler_comp(
            path_doppler, path_markers, path_markers_label)
        print('@@ bbox_doppler:', bbox_doppler)
        print('@@ bbox_markers:', bbox_markers)

        iou = get_iou(bbox_doppler, bbox_markers)
        print('@@ iou:', iou)

        plt = plot_comp(border_img_doppler, border_img_markers, path_doppler, path_markers)
        stem = os.path.splitext(os.path.basename(path_doppler))[0]
        fname = f'{savepath}/comp-doppler-{stem}.jpg'
        plt.savefig(fname, bbox_inches='tight')
        print('@@ saved -', fname)


# >>> m = [1, 2, 3, 4, 5]
# >>> slice_split(m, slice(2, 4))
# ([3, 4], [1, 2, 5])
def slice_split(li_in_const, slice_in):
    li_out = list(li_in_const)  # "copy" the list
    li_out_sliced = li_in_const[slice_in]
    del li_out[slice_in]
    return li_out_sliced, li_out

def create_train_loader(train_ds_path, target_resize, batch_size, workers, with_doppler=False):
    #----!!!!
    from wsdan.net.doppler import get_to_doppler
    dataset_doppler_root = train_ds_path['benign'][0].split('/')[0]
    #print('@@ dataset_doppler_root:', dataset_doppler_root)
    #----!!!!

    train_dataset = ThyroidDataset(
        phase='train',
        dataset=train_ds_path,
        transform=get_transform(target_resize, phase='basic'),
    #==== @@ orig
        with_alpha_channel=False  # if False, it will load image as RGB(3-channel)
    #==== @@ WIP w.r.t. 'digitake/preprocess/thyroid.py'
        # mask_dict=get_to_doppler(dataset_doppler_root) if with_doppler else None,  # !!!!
        # with_alpha_channel=with_doppler  # !!!! TODO debug with `True`
    #====
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True)

    if 0:
        print('@@ show_data_loader(train_loader) -------- ^^')
        _channel, _, _, _ = show_data_loader(train_loader)  # only the first batch shown
        print('@@ show_data_loader(train_loader) -------- $$')

    return train_loader

def create_validate_loader(validate_ds_path, target_resize, batch_size, workers):
    validate_dataset = ThyroidDataset(
        phase='val',
        dataset=validate_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    return DataLoader(
        validate_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)


def kfold_ds_paths_debug_v1():  # hardcoded w.r.t. 'Dataset_train_test_val.zip'
    mix_ds_path  = build_dataset({
        'benign': ['Train/Benign', 'Val/Benign'],
        'malignant': ['Train/Malignant', 'Val/Malignant'],
    }, root='Dataset_train_test_val')  # 30 30
    print("@@ lens trainval_ds_path:", len(mix_ds_path['benign']), len(mix_ds_path['malignant']))

    # fold 0
    _s = slice(0, 10)
    mix_ben_v, mix_ben_t = slice_split(mix_ds_path['benign'], _s)
    mix_mal_v, mix_mal_t = slice_split(mix_ds_path['malignant'], _s)
    t_0_ds_path = {'benign': mix_ben_t, 'malignant': mix_mal_t}
    v_0_ds_path = {'benign': mix_ben_v, 'malignant': mix_mal_v}
    if 0:  # lgtm
        __t_0_ds_path = {'benign': mix_ds_path['benign'][10:], 'malignant': mix_ds_path['malignant'][10:]}
        __v_0_ds_path = {'benign': mix_ds_path['benign'][0:10], 'malignant': mix_ds_path['malignant'][0:10]}
        print(t_0_ds_path, __t_0_ds_path)
        print(v_0_ds_path, __v_0_ds_path)
        exit()

    # fold 1
    _s = slice(10, 20)
    mix_ben_v, mix_ben_t = slice_split(mix_ds_path['benign'], _s)
    mix_mal_v, mix_mal_t = slice_split(mix_ds_path['malignant'], _s)
    t_1_ds_path = {'benign': mix_ben_t, 'malignant': mix_mal_t}
    v_1_ds_path = {'benign': mix_ben_v, 'malignant': mix_mal_v}
    if 0:  # lgtm
        __t_1_ds_path = {'benign': [], 'malignant': []}  # !!!!dummy
        __v_1_ds_path = {'benign': mix_ds_path['benign'][10:20], 'malignant': mix_ds_path['malignant'][10:20]}
        print(t_1_ds_path, __t_1_ds_path)
        print(v_1_ds_path, __v_1_ds_path)
        exit()

    # fold 2
    _s = slice(20, 30)
    mix_ben_v, mix_ben_t = slice_split(mix_ds_path['benign'], _s)
    mix_mal_v, mix_mal_t = slice_split(mix_ds_path['malignant'], _s)
    t_2_ds_path = {'benign': mix_ben_t, 'malignant': mix_mal_t}
    v_2_ds_path = {'benign': mix_ben_v, 'malignant': mix_mal_v}
    if 0:  # lgtm
        __t_2_ds_path = {'benign': mix_ds_path['benign'][0:20], 'malignant': mix_ds_path['malignant'][0:20]}
        __v_2_ds_path = {'benign': mix_ds_path['benign'][20:30], 'malignant': mix_ds_path['malignant'][20:30]}
        print(t_2_ds_path, __t_2_ds_path)
        print(v_2_ds_path, __v_2_ds_path)
        exit()

    out = [(t_0_ds_path, v_0_ds_path), (t_1_ds_path, v_1_ds_path), (t_2_ds_path, v_2_ds_path)]
    for k, tv in enumerate(out):
        print(f"@@ fold: {k}")
        print("@@ lens t_k_ds_path:", len(tv[0]['benign']), len(tv[0]['malignant']))  # 20 20
        print("@@ lens v_k_ds_path:", len(tv[1]['benign']), len(tv[1]['malignant']))  # 10 10
        # print("@@ lens t_k_ds_path:", tv[0]['benign'], tv[0]['malignant'])  # [...] [...]
        # print("@@ lens v_k_ds_path:", tv[1]['benign'], tv[1]['malignant'])  # [...] [...]

    return out


def slice_mix_ds_path(mix_ds_path, slice_v_ben, slice_v_mal):
    mix_ben_vt = slice_split(mix_ds_path['benign'], slice_v_ben)
    mix_mal_vt = slice_split(mix_ds_path['malignant'], slice_v_mal)
    return ({'benign': mix_ben_vt[1], 'malignant': mix_mal_vt[1]},
            {'benign': mix_ben_vt[0], 'malignant': mix_mal_vt[0]})

def kfold_ds_paths_debug_v2():  # hardcoded w.r.t. 'Dataset_train_test_val.zip'
    mix_ds_path  = build_dataset({
        'benign': ['Train/Benign', 'Val/Benign'],
        'malignant': ['Train/Malignant', 'Val/Malignant'],
    }, root='Dataset_train_test_val')  # 30 30
    print("@@ lens trainval_ds_path:", len(mix_ds_path['benign']), len(mix_ds_path['malignant']))

    return [slice_mix_ds_path(mix_ds_path, slice_v)
            for slice_v in (slice(0, 10), slice(10, 20), slice(20, 30))]


def _train(with_doppler, total_epochs, model, ds_paths, savepath, config_doppler=None):
    device = get_device()
    print("@@ device:", device)

    print('@@ with_doppler:', with_doppler)
    print('@@ config_doppler:', config_doppler)
    print('@@ total_epochs:', total_epochs)
    print('@@ model:', model)
    print('@@ savepath:', savepath)

    for k, dsp in ds_paths.items():
        if k in ['train', 'validate', 'kfold']:
            print(f"@@ lens of ds_paths['{k}']:", len(dsp['benign']), len(dsp['malignant']))
        elif k == 'kfold_slices_val':
            print(f"@@ len(ds_paths['{k}']):", len(dsp))
        else:
            raise ValueError(f'unknown ds_paths key: {k}')

    target_resize = 250
    batch_size = 8 #@param ["8", "16", "4", "1"] {type:"raw"}

    number = 4 #@param ["1", "2", "3", "4", "5"] {type:"raw", allow-input: true}

    workers = 2
    print('@@ workers:', workers)

    lr = 0.001 #@param ["0.001", "0.00001"] {type:"raw"}
    lr_ = "lr-1e5" #@param ["lr-1e3", "lr-1e5"]

    run_name = f"{model}_{target_resize}_{batch_size}_{lr_}_n{number}"
    print('@@ run_name:', run_name)

    #

    kfold_ds_path = ds_paths.get('kfold')
    if kfold_ds_path is None:
        print("@@ k-fold is disabled")
        kfold_ds_paths = [(ds_paths['train'], ds_paths['validate'])]
    else:
        print("@@ k-fold is ENABLED")
        #====
        ##kfold_ds_paths = kfold_ds_paths_debug_v1()
        ##kfold_ds_paths = kfold_ds_paths_debug_v2()
        #==== @@
        k = len(ds_paths['kfold_slices_val'])
        print("@@ k:", k)

        #---- ^^ adjust dataset lengths, updating `kfold_ds_path`
        mix_ben_len_truncated = len(kfold_ds_path['benign']) - len(kfold_ds_path['benign']) % k
        mix_mal_len_truncated = len(kfold_ds_path['malignant']) - len(kfold_ds_path['malignant']) % k
        kfold_ds_path['benign'] = kfold_ds_path['benign'][0:mix_ben_len_truncated]
        kfold_ds_path['malignant'] = kfold_ds_path['malignant'][0:mix_mal_len_truncated]

        mix_ben_len, mix_mal_len = len(kfold_ds_path['benign']), len(kfold_ds_path['malignant'])
        assert mix_ben_len % k == 0
        assert mix_mal_len % k == 0
        print("@@ [after truncation] lens of kfold_ds_path:", mix_ben_len, mix_mal_len)
        #---- $$ adjust dataset lengths

        kfold_ds_paths = [slice_mix_ds_path(kfold_ds_path, svb, svm)
                          for (svb, svm) in ds_paths['kfold_slices_val']]
        if 1:  # check
            ##print("@@ kfold_ds_paths:", kfold_ds_paths)
            print("@@ -------- `kfold_ds_paths`, check: ^^")
            for smdp in kfold_ds_paths:
                print("@@ ----")
                print("@@ bt:", len(smdp[0]['benign']))
                print("@@ mt:", len(smdp[0]['malignant']))
                print("@@ bv:", len(smdp[1]['benign']))
                print("@@ mv:", len(smdp[1]['malignant']))
            print("@@ -------- `kfold_ds_paths`, check: $$")
            ##exit()  # !!!!
        #====

    kfold_loaders = [(
        create_train_loader(tv_ds_path[0], target_resize, batch_size, workers, with_doppler),
        create_validate_loader(tv_ds_path[1], target_resize, batch_size, workers))
        for tv_ds_path in kfold_ds_paths]

    #

    num_attention_maps = 32  # @@ cf. 16 in 'main_legacy.py'
    net = WSDAN(num_classes=WSDAN_NUM_CLASSES, M=num_attention_maps, model=model, pretrained=True)
    net.to(device)
    feature_center = torch.zeros(WSDAN_NUM_CLASSES, num_attention_maps * net.num_features).to(device)

    #

    logs = {
        'epoch': 0,
        'train/loss': float("Inf"),
        'val/loss': float("Inf"),
        'train/raw_topk_accuracy': 0.,
        'train/crop_topk_accuracy': 0.,
        'train/drop_topk_accuracy': 0.,
        'val/topk_accuracy': 0.
    }

    learning_rate = logs['lr'] if 'lr' in logs else lr

    opt_type = 'SGD'
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)

    START_EPOCH = 0

    if 0:  # @@
        wandb.init(
            # Set the project where this run will be logged
            # project=f"Wsdan_Thyroid_{total_epochs}epochs_RecheckRemove_Upsampling_v2",
            project=f"Wsdan_Thyroid",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=run_name,
            # Track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "architecture": f"WS-DAN-{model}",
            "optimizer": opt_type,
            "dataset": "Thyroid",
            "train-data-augment": f"{channel}-channel",
            "epochs": f"{total_epochs - START_EPOCH}({START_EPOCH}->{total_epochs})" ,
        })

    #

    print('Start training: Total epochs: {}, Batch size: {}'.format(total_epochs, batch_size))

    ckpt = net_train.train(
        device, net, feature_center, batch_size, kfold_loaders,
        optimizer, scheduler, run_name, logs, START_EPOCH, total_epochs,
        with_doppler=with_doppler, config_doppler=config_doppler, savepath=savepath)
    print('@@ done; ckpt:', ckpt)

    return ckpt


def train(
        total_epochs=TOTAL_EPOCHS_DEFAULT,
        model=MODEL_DEFAULT,
        ds_paths={'train': TRAIN_DS_PATH_DEFAULT, 'validate': VALIDATE_DS_PATH_DEFAULT}):
    return _train(False, total_epochs, model, ds_paths, mk_artifact_dir('demo_train'))


def train_with_doppler(
        total_epochs=TOTAL_EPOCHS_DEFAULT,
        model=MODEL_DEFAULT,
        ds_paths={'train': TRAIN_DS_PATH_DEFAULT, 'validate': VALIDATE_DS_PATH_DEFAULT},
        config_doppler={
            'thresh_isec_in_crop': 0.25,  # default
            #'thresh_isec_in_crop': 0.50,
            #'thresh_isec_in_crop': 0.75,
            #'thresh_force_doppler_in_crop': True,
        }):
    return _train(True, total_epochs, model, ds_paths, mk_artifact_dir('demo_train_with_doppler'),
                  config_doppler=config_doppler)


def test(ckpt, model=MODEL_DEFAULT, ds_path=TEST_DS_PATH_DEFAULT,
        target_resize=250, batch_size=8, num_attention_maps=32, auc=False, tag=''):
    from .utils import show_data_loader
    from .stats import print_scores, print_auc, print_poa

    print("@@ model:", model)
    print("@@ target_resize:", target_resize)
    print("@@ batch_size:", batch_size)

    device = get_device()
    print("@@ device:", device)

    #print('@@ ds_path:', ds_path)
    print("@@ lens ds_path:", len(ds_path['benign']), len(ds_path['malignant']))

    test_dataset = ThyroidDataset(
        phase='test',
        dataset=ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    #@@workers = 2
    workers = 0  # @@
    print('@@ workers:', workers)

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # @@
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    #

    net = WSDAN(num_classes=WSDAN_NUM_CLASSES, M=num_attention_maps, model=model, pretrained=True)
    net.to(device)

    sp = mk_artifact_dir(f'demo_thyroid_test_{tag}')
    results = net_test.test(device, net, batch_size, test_loader, ckpt, savepath=sp)
    # print('@@ results:', results)

    if 1:
        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

    if 1:
        print(f'\n\n@@ ======== print_poa(results)')
        print_poa(results)

    if auc:
        _enable_plot = True  # !!
        print(f'\n\n@@ ======== print_auc(results, plot={_enable_plot}), plot_savepath="{sp}"')
        print_auc(results, len(test_dataset), plot=_enable_plot, plot_savepath=sp)
