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
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#---- @@
from ..plot_if import get_plt, plt_imshow, plt_imshow_tensor, is_colab
plt = get_plt()

from torchvision.transforms import ToPILImage, PILToTensor
transform_to_pil = ToPILImage()
transform_to_tensor = PILToTensor()
#----

def load_data(train_size=5000, test_size=1000):
    print('@@ load_data(): ^^')

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


def get_finetuned(model_name, itos, stoi):
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

    # To use Cifar-10, it needs to be fine tuned again with 10 output classes
    model = ViTForImageClassification.from_pretrained(model_name,
        num_labels=10,
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


from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
class MriDatasetAdapter(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        px, class_index = self.dataset[index]
        return {'img': None, 'label': class_index, 'pixels': px }


def main():

    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)

    transf_inner = Compose([
        Resize(processor.size['height']),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    #==== orig
    if 0:  # orig
        trainds, valds, testds, itos, stoi = load_data()
        num_train_epochs = 3

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
        #debug_skip_training = 1  # !!!!

        preprocess_data(transf_inner, trainds, valds, testds)

        #print(trainds[0]['img'])  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x154163190>
        #print(trainds[0]['label'])  # 4
        #print(trainds[0]['pixels'].shape)  # torch.Size([3, 224, 224])

        exit()  # !!
    #==== @@ MRI: mnist/thyroid
    if 1:  #
        from ..vit.vit_torch import get_mnist_ds_paths, MriDataset

        ds_paths, class_dir_map = get_mnist_ds_paths(debug=True)
        #ds_paths, class_dir_map = get_thyroid_ds_paths(debug=True)  # todo

        itos = dict((i, v) for i, (k, v) in enumerate(sorted(class_dir_map.items())))
        stoi = dict((v, i) for i, (k, v) in enumerate(sorted(class_dir_map.items())))

        transf = lambda pil_img : transf_inner(pil_img.convert('RGB'))
        train_set = MriDataset(
            phase='finetune_train',
            dataset=ds_paths['train'],
            transform=transf)
        test_set = MriDataset(
            phase='finetune_test',
            dataset=ds_paths['test'],
            transform=transf)

        if 1:  # debug
            px, class_index = train_set[0]
            print(px.shape, class_index)  # torch.Size([3, 224, 224]) 0
            px, class_index = train_set[5922]
            print(px.shape, class_index)  # torch.Size([3, 224, 224]) 0
            px, class_index = train_set[5923]
            print(px.shape, class_index)  # torch.Size([3, 224, 224]) 1

        # [ ] {train,test}_set --> {train,val,test}ds
        #====
        trainds = MriDatasetAdapter(train_set)
        print(len(trainds))  # 60000

        for i in range(5923-4, 5923+4):
            x = trainds[i]
            print(i, x['img'], x['label'], x['pixels'].shape)

        valds = trainds  # !!!!
        testds = trainds  # !!!!
        num_train_epochs = 1  # !!!!
        debug_skip_training = 0  # !!!!

        #exit()  # !!!!


    #@@ ??
    #!pip show accelerate

    trainer = get_trainer(
        get_finetuned(model_name, itos, stoi),
        TrainingArguments(
            f"test-cifar-10",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=4,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='logs',
            remove_unused_columns=False,
        ),
        processor, trainds, valds)

    """### Training the model for fine tuning"""

    if 0:  #@@
        pass
        # Commented out IPython magic to ensure Python compatibility.
        # %load_ext tensorboard
        # %tensorboard --logdir logs/

    if not debug_skip_training:
        print('@@ calling `trainer.train()`')
        trainer.train()

    """### Evaluation"""

    print('@@ calling `trainer.predict(testds)`')
    outputs = trainer.predict(testds)
    print(outputs.metrics)
    """ 250 <-- 1000 / 4 (test_size / args.per_device_eval_batch_size)
100%|██████████| 250/250 [15:11<00:00,  3.65s/it]
{'test_loss': 2.4680304527282715, 'test_model_preparation_time': 0.0091, 'test_accuracy': 0.075, 'test_runtime': 914.5631, 'test_samples_per_second': 1.093, 'test_steps_per_second': 0.273}
    """

    if 1:  # confusion matrix
        itos[np.argmax(outputs.predictions[0])], itos[outputs.label_ids[0]]

        y_true = outputs.label_ids
        y_pred = outputs.predictions.argmax(1)

        labels = trainds.features['label'].names
        cm = confusion_matrix(y_true, y_pred)

        fname = 'confusion_matrix.png'
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        print(f'@@ saving {fname}')
        disp.plot(xticks_rotation=45).figure_.savefig(fname)
        if is_colab():
            plt_imshow(plt, fname)


if __name__ == "__main__":
    main(train_set, test_set)
