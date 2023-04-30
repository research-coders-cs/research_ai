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


def _train(with_doppler, total_epochs, model, train_ds_path, validate_ds_path, savepath):
    device = get_device()
    print("@@ device:", device)

    print('@@ with_doppler:', with_doppler)
    print('@@ total_epochs:', total_epochs)
    print('@@ model:', model)
    print('@@ savepath:', savepath)

    #print('@@ train_ds_path:', train_ds_path)
    print("@@ lens train_ds_path:", len(train_ds_path['benign']), len(train_ds_path['malignant']))

    #print('@@ validate_ds_path:', validate_ds_path)
    print("@@ lens validate_ds_path:", len(validate_ds_path['benign']), len(validate_ds_path['malignant']))

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

    from wsdan.net.doppler import to_doppler  # !!!!

    train_dataset = ThyroidDataset(
        phase='train',
        dataset=train_ds_path,
        transform=get_transform(target_resize, phase='basic'),
    #==== @@ orig
        with_alpha_channel=False  # if False, it will load image as RGB(3-channel)
    #==== @@ WIP w.r.t. 'digitake/preprocess/thyroid.py'
        # mask_dict=to_doppler if with_doppler else None,  # !!!!
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

    #

    validate_dataset = ThyroidDataset(
        phase='val',
        dataset=validate_ds_path,
        transform=get_transform(target_resize, phase='basic'),
        with_alpha_channel=False)

    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

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

    print('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'
        .format(total_epochs, batch_size, len(train_dataset), len(validate_dataset)))

    ckpt = net_train.train(
        device, net, feature_center, batch_size, train_loader, validate_loader,
        optimizer, scheduler, run_name, logs, START_EPOCH, total_epochs,
        with_doppler=with_doppler, savepath=savepath)
    print('@@ done; ckpt:', ckpt)

    return ckpt


def train(
        total_epochs=TOTAL_EPOCHS_DEFAULT,
        model=MODEL_DEFAULT,
        train_ds_path=TRAIN_DS_PATH_DEFAULT,
        validate_ds_path=VALIDATE_DS_PATH_DEFAULT):
    return _train(False, total_epochs, model, train_ds_path, validate_ds_path,
        mk_artifact_dir('demo_train'))


def train_with_doppler(
        total_epochs=TOTAL_EPOCHS_DEFAULT,
        model=MODEL_DEFAULT,
        train_ds_path=TRAIN_DS_PATH_DEFAULT,
        validate_ds_path=VALIDATE_DS_PATH_DEFAULT):
    return _train(True, total_epochs, model, train_ds_path, validate_ds_path,
        mk_artifact_dir('demo_train_with_doppler'))


def test(ckpt, model=MODEL_DEFAULT, ds_path=TEST_DS_PATH_DEFAULT,
        target_resize=250, batch_size=8, num_attention_maps=32):
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

    results = net_test.test(device, net, batch_size, test_loader, ckpt,
        savepath=mk_artifact_dir('demo_thyroid_test'))
    # print('@@ results:', results)

    if 1:
        print('\n\n@@ ======== print_scores(results)')
        print_scores(results)

    if 0:
        _enable_plot = 0  # @@
        print(f'\n\n@@ ======== print_auc(results, enable_plot={_enable_plot})')
        print_auc(results, len(test_dataset), enable_plot=_enable_plot)

    if 1:
        print(f'\n\n@@ ======== print_poa(results)')
        print_poa(results)