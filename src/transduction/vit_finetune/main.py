"""
adapted from -- research_mri/transduction/finetune/try_vision_transformers_hugging_face_fine_tuning_cifar10_pytorch.py
"""

# https://huggingface.co/docs/transformers/en/installation
# For CPU-support only, you can conveniently install 🤗 Transformers and a deep learning library in one line. For example, install 🤗 Transformers and PyTorch with:
# pip install 'transformers[torch]'

"""@@
$ pipenv run python3 -m pip install 'transformers[torch]'
Successfully installed accelerate-1.0.0 filelock-3.16.1 fsspec-2024.9.0 huggingface-hub-0.25.1 regex-2024.9.11 safetensors-0.4.5 tokenizers-0.20.0 transformers-4.45.2

$ pipenv run python3 -m pip install datasets
Successfully installed datasets-3.0.1 dill-0.3.8 fsspec-2024.6.1 multiprocess-0.70.16 pyarrow-17.0.0 xxhash-3.5.0

$ pipenv run python3 -m pip install scikit-learn
Successfully installed joblib-1.4.2 scikit-learn-1.5.2 scipy-1.14.1 threadpoolctl-3.5.0
"""


import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

#---- @@
from ..plot_if import get_plt, plt_imshow, plt_imshow_tensor, get_confusion_matrix
plt = get_plt()

from torchvision.transforms import ToPILImage, PILToTensor
transform_to_pil = ToPILImage()
transform_to_tensor = PILToTensor()

from .attention import plot_attention, plot_attention_heads
from ..vit.vit_torch import MriDataset, plot_vit_patches
import cv2
#----

def load_data(train_size=5000, test_size=1000):
    print('@@ load_data(): ^^')

    from datasets import load_dataset  # cifar10
    trainds, testds = load_dataset("cifar10", split=[f"train[:{train_size}]", f"test[:{test_size}]"])

    splits = trainds.train_test_split(test_size=0.1)
    trainds = splits['train']  # 90%
    valds = splits['test']  # 10%

    #print(type(trainds))  # <class 'datasets.arrow_dataset.Dataset'>
    #print(trainds.features, trainds.num_rows, trainds[0])
    # {  'img': Image(mode=None, decode=True, id=None),
    #    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], id=None)
    # }
    # 9
    # {  'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x1489A9900>,
    #    'label': 6
    # }

    itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
    stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))
    ##print(itos, stoi)
    # {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    if 0:
        img, lab = trainds[0]['img'], itos[trainds[0]['label']]
        ##print(lab)  # truck
        ##print(img.size)  # (32, 32)

        #img  # colab only
        #print(type(img))  # <class 'PIL.PngImagePlugin.PngImageFile'>
        plt_imshow_tensor(plt, transform_to_tensor(img))

    return trainds, valds, testds, itos, stoi


def preprocess_data(transf_inner, trainds, valds, testds):

    """### Preprocessing Data"""

    def transf(arg):
        arg['pixels'] = [transf_inner(image.convert('RGB')) for image in arg['img']]
        return arg

    trainds.set_transform(transf)
    valds.set_transform(transf)
    testds.set_transform(transf)

    if 1:  # !!
        print(trainds[0].keys())  # dict_keys(['img', 'label', 'pixels'])

        img = trainds[0]['img']
        print(img)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x14A383070>

        px = trainds[0]['pixels']
        print(px.shape)  # torch.Size([3, 224, 224])

        print(torch.min(px), torch.max(px))  # tensor(-0.8745) tensor(1.)
        px = (px+1)/2
        print(torch.min(px), torch.max(px))  # tensor(0.0627) tensor(1.)

        plt_imshow(plt, img)  # orig
        plt_imshow_tensor(plt, px)  # preprocessed
        #plt_imshow(plt, transform_to_pil(px))  # preprocessed, the same

        #exit()  # !!


def get_finetuned(model_name, class_names_sorted):
    print('@@ get_finetuned(): ^^')

    """### Model - Fine Tuning"""

    model_orig = ViTForImageClassification.from_pretrained(model_name)
    print('get_finetuned(): [before] ', model_orig.classifier)
    # The google/vit-base-patch16-224 model is originally fine tuned on imagenet-1K with 1000 output classes

    if 0:
        print(model_orig.config)
        """
        { ...
            "yurt": 915,
            "zebra": 340,
            "zucchini, courgette": 939
          },
          "layer_norm_eps": 1e-12,
          "model_type": "vit",
          "num_attention_heads": 12,
          "num_channels": 3,
          "num_hidden_layers": 12,
          "patch_size": 16,
          "qkv_bias": true,
          "transformers_version": "4.45.2"
        }
        """

    itos = dict((i, k) for i, k in enumerate(class_names_sorted))
    stoi = dict((k, i) for i, k in enumerate(class_names_sorted))

    # To use Cifar-10, it needs to be fine tuned again with 10 output classes
    model = ViTForImageClassification.from_pretrained(model_name,
        attn_implementation="eager",  # @@ https://discuss.huggingface.co/t/attentions-not-returned-from-transformers-vit-model-when-using-output-attentions-true/91203/4
        #@@num_labels=10,
        num_labels=len(itos.keys()),  # @@
        ignore_mismatched_sizes=True,
        id2label=itos,
        label2id=stoi)
    """
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized because the shapes did not match:
- classifier.bias: found shape torch.Size([1000]) in the checkpoint and torch.Size([10]) in the model instantiated
- classifier.weight: found shape torch.Size([1000, 768]) in the checkpoint and torch.Size([10, 768]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    """
    print('get_finetuned(): [after] ', model.classifier)
    if 1:  # debug
        print('get_finetuned(): model.config:', model.config)
        ##exit()

    return model


def get_trainer(model, args, processor, trainds, valds):

    print(f'@@ get_trainer(): args.per_device_train_batch_size={args.per_device_train_batch_size}')
    print(f'@@ get_trainer(): args.per_device_eval_batch_size={args.per_device_eval_batch_size}')
    print(f'@@ get_trainer(): args.num_train_epochs={args.num_train_epochs}')

    def collate_fn(examples):
        pixels = torch.stack([example["pixels"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixels, "labels": labels}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    trainer = Trainer(
        model,
        args,
        train_dataset=trainds,
        eval_dataset=valds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    return trainer


from torch.utils.data import Dataset, random_split
class MriDatasetAdapter(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        px, class_index, extra = self.dataset[index]
        return {'img': extra['path'], 'label': class_index, 'pixels': px }


from ..vit.bs1_atten import Bs1Atten
def verify_attentions(model, testds, verify_sample_size=-1, y_true=None, y_pred=None,
                      ckpt_file=None, save_dir='inference', class_names_sorted=None,
                      mri_ch=250, mri_rh=80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gen_finetune():
        for idx, x in enumerate(testds):
            input = x['pixels']  # torch.Size([3, 224, 224])
            input_path = x['img']

            outputs = model(
                input.to(device).unsqueeze(0),  # --> [1, 3, 224, 224] ([batch_size, channels, height, width])
                output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions
            input = input.cpu()[0, :, :]  # --> torch.Size([224, 224])
            yield idx, input, input_path, logits, attentions

    def gen_arch():
        for idx, x in enumerate(testds):
            input, _, extra = x
            input_path = extra['path'][0]

            logits, attentions = model(
                input.to(device),  # torch.Size([1, 3, 224, 224])
                output_attentions=True)
            input = input.cpu()[0, 0, :, :]  # --> torch.Size([224, 224])
            yield idx, input, input_path, logits, attentions

    #

    model_class = type(model).__name__
    print('@@ model_class:', model_class)
    if model_class == 'ViTForImageClassification':
        gen = gen_finetune
    elif model_class == 'CustomViT':
        gen = gen_arch
    else:
        raise ValueError(f'Unsupported model_class: {model_class}')

    #

    for idx, input, input_path, logits, attentions in gen():
        if verify_sample_size >= 0 and verify_sample_size == idx:
            break

        att_mat = torch.cat(attentions).cpu()  # torch.Size([12, 12, 197, 197]) [num_hidden_layers, num_heads, seq_len, seq_len]

        if 0:
            print('@@ logits:', logits)
            print('@@ type(attentions):', type(attentions))  # <class 'tuple'>
            for i, attn in enumerate(attentions):
                print(f'@@ attn[{i}]: {attn.shape}')  # torch.Size([1, 12, 197, 197])
                # [batch_size, num_heads, seq_len, seq_len]

        #---- Resolve `im_orig`

        #print(f'@@ testds[{idx}]: path={input_path}')
        #==== NG -- "works" for only PNG
        # im_input = plt.imread(input_path.split('?')[0])  # ndarray
        #==== OK -- the pixel data is identical whether the source was PNG or JPG
        img = Image.open(input_path.split('?')[0]).convert("RGB")  # always RGB, no alpha
        im_input = np.asarray(img, dtype=np.float32) / 255.0  # always float [0, 1]
        #====
        #plt_imshow(plt, im_input)

        erica_mode = 'erica=' in input_path
        if erica_mode:
            im_erica_l, im_erica_r = MriDataset.erica_crop_im(im_input, ch=mri_ch, rh=mri_rh)
            if 'erica=l' in input_path:
                im_input = im_erica_l
            elif 'erica=r' in input_path:
                im_input = im_erica_r

        im_orig = cv2.resize(im_input, input.shape)
        #print('@@ im_orig.shape:', im_orig.shape)  # (224, 224, 3)

        #---- Compute heatmaps

        # averaged across all attention heads
        im_heatmap, im_mask = Bs1Atten.compute_heatmap(
            input, att_mat, idx, i_head=None, im_orig=im_orig)

        num_heads = att_mat.shape[1]
        heatmaps_headwise = [ Bs1Atten.compute_heatmap(
            input, att_mat, idx, i_head=i_head, im_orig=im_orig)[0] for i_head in range(num_heads) ]

        #---- Display inference info

        has_yt = y_true is not None
        has_yp = y_pred is not None
        yt = y_true[idx] if has_yt else None
        yp = y_pred[idx] if has_yp else None

        label_yt, label_yp, result, result_tf = None, None, None, None
        if has_yt and has_yp:
            #---- labels/result
            has_labels = class_names_sorted is not None
            label_yt = class_names_sorted[yt] if has_labels else '-'
            label_yp = class_names_sorted[yp] if has_labels else '-'
            result = '✅' if yt == yp else '❌'
            result_tf = yt == yp

            #---- logits -> confidence
            # print('@@ logits:', logits)
            yp_via_logits = logits.argmax(-1)[0].item()
            assert yp == yp_via_logits

            confidence = None
            if has_labels:
                percents = (torch.softmax(logits, dim=-1) * 100).squeeze().tolist()
                sorted_preds = sorted(zip(class_names_sorted, percents), key=lambda x: x[1], reverse=True)
                # print(sorted_preds)
                confidence = [ f"{name}: {percent:.2f}%" for name, percent in sorted_preds ]

            print(f"testds[{idx}]: {result} ytrue: {yt} (={label_yt}) ypred: {yp} (={label_yp}) confidence: {confidence}")

        #-- info -- basic
        title = (f'testds[{idx}] | attention_mask | attention_ave (of {num_heads} heads)\n'
                 f'(path: {input_path})\n'
                 f'(ViT model: {ckpt_file})\n'
                 f'(ytrue: {yt} (={label_yt}) ypred: {yp} (={label_yp}) result: {result_tf})')
        ims = [im_erica_l, im_erica_r, im_mask, im_heatmap] if erica_mode\
            else [im_orig, im_mask, im_heatmap]
        plot_attention(ims, title, f'{save_dir}/info_testds_{idx}_result_{result_tf}.png')

        #-- info -- attention heads
        title = (f'testds[{idx}] | {num_heads} attention heads\n'
            f'(ViT model: {ckpt_file})')
        plot_attention_heads(heatmaps_headwise, im_heatmap, title,
            f'{save_dir}/info_testds_{idx}_attention_heads.png')

        #-- info -- attention ave
        # TODO !!!!!!!!
        #     f'{save_dir}/info_testds_{idx}_attention_ave.png')

        #-- info -- ViT patches
        if 0:  # !! experimental
            title = f'testds[{idx}] | path: {input_path}'
            plot_vit_patches(input_path, title,
                f'{save_dir}/info_testds_{idx}_vit_patches.png')
            #---- fixme colab warnings [ ]
            # Verifying first 4 samples of 100
            # @@ model_class: CustomViT
            #
            # /usr/local/lib/python3.13/dist-packages/transduction/vit_finetune/attention.py:82: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:253.)
            #   mask_stacked = torch.tensor([mask[:,:]], dtype=torch.float32)


def debug_print_dat(dat):
    print(dat)
    #plt_imshow_tensor(plt, dat[0])  # transformed, normalized to the range of -1 to 1

def main():

    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)

    transf_inner = Compose([
        Resize((processor.size['height'], processor.size['width'])),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    #==== orig
    if 0:  # orig
        trainds, valds, testds, itos, stoi = load_data()
        num_train_epochs = 3
        class_names_sorted = sorted(stoi.keys())

        ##print(trainds, valds, testds)
        # Dataset({
        #     features: ['img', 'label'],
        #     num_rows: 4500
        # }) Dataset({
        #     features: ['img', 'label'],
        #     num_rows: 500
        # }) Dataset({
        #     features: ['img', 'label'],
        #     num_rows: 1000
        # })

        preprocess_data(transf_inner, trainds, valds, testds)
    #==== @@
    if 0:  # debug
        trainds, valds, testds, itos, stoi = load_data(train_size=10, test_size=20)
        num_train_epochs = 1  # !!
        class_names_sorted = sorted(stoi.keys())

        preprocess_data(transf_inner, trainds, valds, testds)

        #print(trainds[0]['img'])  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x154163190>
        #print(trainds[0]['label'])  # 4
        #print(trainds[0]['pixels'].shape)  # torch.Size([3, 224, 224])

        #exit()  # !!
    #==== @@ MRI: mnist/thyroid/mri
    if 1:  # !!
        from ..vit.vit_torch import stat_ds_paths, build_dataset
        from ..vit.vit_torch import get_mnist_ds_paths, get_thyroid_ds_paths, get_mri_ds_paths

        transf = lambda pil_img : transf_inner(pil_img)  # default `transf`

        #ds_paths, class_names_sorted = get_mnist_ds_paths(debug=True)
        #ds_paths, class_names_sorted = get_thyroid_ds_paths('ttv', debug=True)
        #ds_paths, class_names_sorted = get_thyroid_ds_paths('100g', debug=True)
        if 0:  # thyroid colab
            ds_paths, class_names_sorted = {
                'train': build_dataset({
                    'benign': ['Markers_Train_Remove_Markers/Benign_Remove/train', 'Markers_Train_Remove_Markers/Benign_Remove/validate'],
                    'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/train', 'Markers_Train_Remove_Markers/Malignant_Remove/validate'],
                }, root='Dataset_doppler_100g'),
                #
                #
                #==== for '100g'
                'test': build_dataset({
                    'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
                    'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
                }, root='Dataset_doppler_100g'),
                #====
                # 'test': build_dataset({
                #     'benign': ['test26/Benign'],
                #     'malignant': ['test26/Malignant'],
                # }, root='siriraj_original_Testset_26'),
                #==== for 'extra'
                # 'test': build_dataset({
                #     'benign': ['test'],  # fixme !!!!
                #     'malignant': [],  # fixme !!!!
                # }, root='thyroid_inference_extra'),
            }, ['benign', 'malignant']
        if 1:  # mri-erica
            #ds_paths, class_names_sorted = get_mri_ds_paths('debug')
            ds_paths, class_names_sorted = get_mri_ds_paths('erica')

            stat_ds_paths(ds_paths)

            # Update `transf`
            transf = lambda pil_img, idx_mri_left_right : transf_inner(
                MriDataset.erica_crop_pil(pil_img, idx_mri_left_right))


        # Build: {train,test}_set

        train_set = MriDataset(
            dataset=ds_paths['train'],
            transform=transf)
        test_set = MriDataset(
            dataset=ds_paths['test'],
            transform=transf)

        if 0:  # ok
            debug_print_dat(train_set[0])
            debug_print_dat(train_set[1])

            # debug_print_dat(train_set[80])  # datasets_mri/50-001/sub-ADNI002S0559_ses-M012/mta_erica_sub-ADNI002S0559_ses-M012_120.png?erica=r
            # debug_print_dat(train_set[81])  # datasets_mri/50-001/sub-ADNI002S0559_ses-M012/mta_erica_sub-ADNI002S0559_ses-M012_135.png?erica=r
            exit()  # !!

        # Convert: {train,test}_set --> {train,val,test}ds

        if 0:
            len_val = 6000  # ~10%
            len_train = len(train_set) - len_val  # ~90%
            train_set_train, train_set_val = random_split(train_set, [len_train, len_val])
        elif 0:  # !! mnist; CPU experiments
            #train_set_train, train_set_val, _ = random_split(train_set, [90, 10, len(train_set)-100])  # cpu ~3 min
            train_set_train, train_set_val, _ = random_split(train_set, [180, 20, len(train_set)-200])  # cpu ~6 min

            #test_set, _ = random_split(test_set, [40, len(test_set) - 40])
            test_set, _ = random_split(test_set, [10, len(test_set) - 10])
        elif 0:  # thyroid
            #train_set_train, train_set_val = random_split(train_set, [55, 5])  # for 'ttv'
            train_set_train, train_set_val = random_split(train_set, [700, 50])  # for '100g'
        elif 1:  # mri-erica
            train_set_train, train_set_val, test_set, _ = random_split(train_set, [80, 10, 10, len(train_set)-100])
        else:
            pass

        trainds = MriDatasetAdapter(train_set_train)
        valds = MriDatasetAdapter(train_set_val)
        testds = MriDatasetAdapter(test_set)

        print('len({train,val,test}ds):', len(trainds), len(valds), len(testds))  # eg. mnist: 54000 6000 1280

        if 0:  # debug, LGTM (randomised)
            for i in range(0, 8):
                x = trainds[i]
                print(i, x['img'], x['label'], x['pixels'].shape)

        #num_train_epochs = 1  # !! cifar10 orig -> 3
        num_train_epochs = 1  # !! try: mnist full; ~ 30 mins with T4
        #num_train_epochs = 10  # !! try: thyroid 100g
        #exit()  # !!!!

    #

    model = get_finetuned(model_name, class_names_sorted)

    """ **
    The Trainer uses trainds (training dataset) and valds (validation dataset)
    to compute training loss and evaluate metrics (e.g., accuracy) on the validation
    set after each epoch, as specified by evaluation_strategy="epoch". This allows
    us to monitor model performance on valds and use metric_for_best_model="accuracy"
    to save the best model based on validation accuracy.
    """

    trainer = get_trainer(
        model,
        TrainingArguments(
            f"output_trainer_finetune",  # @@
            save_strategy="epoch",
            evaluation_strategy="epoch",  # **
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=4,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,  # **
            metric_for_best_model="accuracy",  # **
            logging_dir='logs',
            remove_unused_columns=False,
            report_to="none",  # @@ https://discuss.huggingface.co/t/how-to-turn-wandb-off-in-trainer/6237/3
        ),
        processor, trainds, valds)

    #

    from ..vit.vit_torch import _save_ckpt, _load_ckpt
    #ckpt_saved = 'foo.ckpt'
    #ckpt_saved = 'foo_debug_eps1.ckpt'
    #ckpt_saved = 'mnist_trained_full.ckpt'
    #----
    #ckpt_saved = 'thyroid_skip_finetune.ckpt'
    #ckpt_saved = 'thyroid_trained_eps8_full.ckpt'
    #----
    ckpt_saved = 'mri_trained_eps1_debug.ckpt'

    if 1:
        print('@@ using `ckpt_saved`:', ckpt_saved)
        model_dict = _load_ckpt(model, ckpt_saved)
        model.load_state_dict(model_dict)
    else:
        print('@@ calling `trainer.train()`')
        trainer.train()
        _save_ckpt(model, ckpt_saved)

    #

    print('@@ calling `trainer.predict(testds)`')
    outputs = trainer.predict(testds)

    print('@@ metrics:', outputs.metrics)

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    #print(y_true, y_pred)  # e.g. [2 1 0 0 2 0 1 1 0 0] [0 0 0 1 0 1 0 0 0 0]

    if 1:
        get_confusion_matrix(y_true, y_pred, class_names_sorted)

    if 1:
        import os

        attn_dir = 'inference_attention_finetune'
        if not os.path.exists(attn_dir):
            os.makedirs(attn_dir, exist_ok=True)

        verify_attentions(model, testds,
                          y_true=y_true, y_pred=y_pred,
                          ckpt_file=ckpt_saved, save_dir=attn_dir)


if __name__ == "__main__":
    main()
