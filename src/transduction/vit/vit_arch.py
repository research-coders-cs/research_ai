
# pip install torch transformers datasets

##

import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .bs1_atten import Bs1Atten
from ..plot_if import get_confusion_matrix
from ..vit.vit_torch import stat_ds_paths, get_mri_ds_paths, MriDataset,\
    ls_ds_path, random_split_ds_path

"""
It’s absolutely possible to customize the model architecture while retaining the
pretrained benefits of "google/vit-base-patch16-224". The key is to strategically modify
the architecture while preserving and leveraging the pretrained weights where possible.
Since you’re working with MNIST (1-channel, scaled to 224x224) and want to debug attention
masks during inference, I’ll show you how to customize the architecture, transfer the
relevant pretrained weights, and maintain the ability to inspect attention.

# Strategy for Customization with Pretrained Benefits

1. Preserve Core Components: Keep the pretrained transformer encoder (with its 12 layers, 768 hidden size, 12 heads) to retain ImageNet-learned representations.

2. Adapt Input Handling: Customize the patch embedding layer for 1-channel MNIST while optionally initializing it with pretrained weights.

3. Modify Other Parts: Adjust the classifier head or add layers (e.g., for debugging) while fine-tuning.

4. Transfer Weights: Load pretrained weights into compatible parts of your custom model.

Here’s how we can do this:

# Customized Model Architecture

Let’s create a custom Vision Transformer that:

- Uses 1-channel input (MNIST-specific).

- Keeps the pretrained transformer encoder.

- Allows attention mask debugging.

- Optionally modifies the number of layers or heads (e.g., reducing to 4 layers like your earlier setup).
"""

from transformers import ViTModel, ViTConfig, ViTImageProcessor

class CustomViT(nn.Module):
    pretrained_model_name = "google/vit-base-patch16-224"

    @staticmethod
    def get_image_processor():
        return ViTImageProcessor.from_pretrained(CustomViT.pretrained_model_name)

    def __init__(self, num_classes=10, num_hidden_layers=None):
        super().__init__()

        model_name = CustomViT.pretrained_model_name

        self.pretrained_config = ViTConfig.from_pretrained(model_name)
        if num_hidden_layers is not None:
            self.pretrained_config.num_hidden_layers = num_hidden_layers

        pretrained_vit = ViTModel.from_pretrained(
            model_name, output_attentions=True, attn_implementation="eager")

        # Channel adapter for 1-to-3 channel conversion
        # Why
        # - Full Pretrained Weights: The patch_embeddings layer retains all RGB-specific information from "google/vit-base-patch16-224", avoiding the loss of detail from averaging.
        # - Learnable Adapter: The channel_adapter learns during fine-tuning to optimally map MNIST’s grayscale to a 3-channel space that the pretrained patch_embeddings can process, enhancing feature extraction.
        # - Consistency: Matches the original ViT input pipeline more closely, leveraging ImageNet-pretrained capabilities.
        self.channel_adapter = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            bias=False
        )

        # Use pretrained patch embeddings
        self.patch_embeddings = pretrained_vit.embeddings.patch_embeddings

        # Clone pretrained CLS token
        self.cls_token = nn.Parameter(pretrained_vit.embeddings.cls_token.clone())

        self.encoder = pretrained_vit.encoder
        if num_hidden_layers is not None and num_hidden_layers < len(self.encoder.layer):
            self.encoder.layer = nn.ModuleList(self.encoder.layer[:num_hidden_layers])

        self.classifier = nn.Linear(self.pretrained_config.hidden_size, num_classes)

    # override
    def forward(self, pixel_values, output_attentions=False):
        batch_size = pixel_values.shape[0]
        channel_size = pixel_values.shape[1]

        if channel_size == 1:
            pixel_values = self.channel_adapter(pixel_values)  # (batch, 1, 224, 224) -> (batch, 3, 224, 224)

        patch_embeds = self.patch_embeddings(pixel_values)  # Already (batch, 196, 768)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, 768)
        embeddings = torch.cat((cls_tokens, patch_embeds), dim=1)  # (batch, 197, 768)

        encoder_outputs = self.encoder(embeddings, output_attentions=output_attentions)
        sequence_output = encoder_outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)

        if output_attentions:
            return logits, encoder_outputs.attentions
        return logits

    def custom_train(self, device, optimizer, loss_fn, epoch_n, train_dataloader,
                     val_dataloader=None, save_best_val_model=True, save_path="best_val_model.pth"):
        best_val_accuracy = 0.0
        best_model_state = None

        for epoch in range(epoch_n):
            # Training phase
            self.train()  # Set model to training mode
            epoch_train_loss = 0.0
            for batch in train_dataloader:
                pixels, labels, _extra = batch
                pixels, labels = pixels.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(pixels)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            avg_train_loss = epoch_train_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epoch_n}, Average Training Loss: {avg_train_loss:.4f}")

            # Validation phase (if val_dataloader is provided)
            if val_dataloader is not None:
                self.eval()  # Set model to evaluation mode; disable dropout and batch normalization updates.
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        pixels, labels, _extra = batch
                        pixels, labels = pixels.to(device), labels.to(device)
                        outputs = self(pixels)
                        loss = loss_fn(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                avg_val_loss = val_loss / len(val_dataloader)
                val_accuracy = 100 * correct / total
                print(f"Epoch {epoch+1}/{epoch_n}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                # Save best model based on validation accuracy
                if save_best_val_model and val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_state = self.state_dict()
                    torch.save(best_model_state, save_path)
                    print(f"🔥 Saved best model with Validation Accuracy: {best_val_accuracy:.2f}% (to {save_path})")

        # Load best model at the end of training; mimicking `load_best_model_at_end=True`
        if save_best_val_model and best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with Validation Accuracy: {best_val_accuracy:.2f}%")


    def custom_test(self, device, test_dataloader):
        predicted_cat = torch.tensor([], dtype=torch.long)
        labels_cat = torch.tensor([], dtype=torch.long)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                pixels, labels, _extra = batch
                pixels, labels = pixels.to(device), labels.to(device)
                outputs = self(pixels)  # with shape [batch_size, num_classes]
                _, predicted = torch.max(outputs, 1)

                predicted = predicted.cpu()
                labels = labels.cpu()
                predicted_cat = torch.cat((predicted_cat, predicted), dim=0)
                labels_cat = torch.cat((labels_cat, labels), dim=0)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

        if 0:
            print(f'@@ (len={len(predicted_cat)}) predicted_cat:', predicted_cat)
            print(f'@@ (len={len(labels_cat)}) labels_cat:', labels_cat)

        return labels_cat, predicted_cat


class CustomMnistDataset(Dataset):
    def __init__(self, ds, transform=None):
        # @@ ds: Dataset({
        #     features: ['image', 'label'],
        #     num_rows: 60000
        # })
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]['image']
        label = self.ds[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

#----^^ commons: CustomMriDataset / CustomDaoDataset
def get_histogram(self_ds, self_len, class_names_sorted):
    hg = {label: 0 for label in class_names_sorted}
    for item in self_ds:
        label = item[2]['label']
        hg[label] += 1

    total = sum(hg.values())
    if total != self_len:
        print(f'histogram WARNING: sum(={total}) and len(={self_len}) do not agree!')
    return f'total={total} {hg}'


def get_transform():
    processor = CustomViT.get_image_processor()
    return transforms.Compose([
        transforms.Resize((processor.size['height'], processor.size['width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
#----$$


class CustomMriDataset(Dataset):
    def __init__(self, ds_path, ch=230, rh=80):
        self.ds = MriDataset(
            dataset=ds_path,
            transform=CustomMriDataset.get_transform(ch, rh))
        self.ds_path = ds_path
        self.ch = ch
        self.rh = rh

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]  # pixels, class_index, extra

    def random_split(self, li):
        return [CustomMriDataset(dsp, ch=self.ch, rh=self.rh)
                for dsp in random_split_ds_path(self.ds_path, li)]

    def get_histogram(self, class_names_sorted):
        return get_histogram(self.ds, len(self), class_names_sorted)

    @staticmethod
    def get_transform(ch, rh):
        transf_inner = get_transform()
        return lambda pil_img, idx_mri_left_right : transf_inner(
            MriDataset.erica_crop_pil(pil_img, idx_mri_left_right, ch=ch, rh=rh))


class CustomDaoDataset(Dataset):
    def __init__(self, ds_path):
        self.ds = MriDataset(  # @@ reuse for Dao ok
            dataset=ds_path,
            transform=get_transform())
        self.ds_path = ds_path

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]  # pixels, class_index, extra

    def random_split(self, li):
        return [CustomDaoDataset(dsp)
                for dsp in random_split_ds_path(self.ds_path, li)]

    def get_histogram(self, class_names_sorted):
        return get_histogram(self.ds, len(self), class_names_sorted)


def main():

    print('@@ vit arch !!')

    if 1:  # debug
        from PIL import Image
        import numpy as np
        import cv2
        from ..vit.vit_torch import patchify_mri, patches_show, patches_savefig
        from ..plot_if import get_plt  #, plt_imshow, plt_imshow_tensor
        plt = get_plt()

        input_path = 'datasets_dao/mars/mars_1111.png'

        # PNG/JPG compat
        img = Image.open(input_path).convert("RGB")  # always RGB, no alpha
        im = np.asarray(img, dtype=np.float32) / 255.0  # always float [0, 1]
        im = cv2.resize(im, (224, 224))
        print('@@ im.shape:', im.shape)  # (224, 224, 3)

        if len(im.shape) == 2:
            tens = torch.tensor([im[:,:]], dtype=torch.float32)  # grayscale
        else:
            tens = torch.tensor([im[:,:,0]], dtype=torch.float32)  # extract R channel as tensor
        print('@@ tens.shape:', tens.shape)  # torch.Size([1, 224, 224])

        tens_stacked = torch.stack([tens], dim=0)
        print('@@ tens_stacked.shape:', tens_stacked.shape)  # torch.Size([1, 1, 224, 224])

        patches = patchify_mri(tens_stacked, (14, 14))
        print('@@ patches.shape:', patches.shape)  # torch.Size([1, 196, 256])

        idx = 999

        idx_stack = 0
        #patches_show(plt, patches, idx_stack, (14, 14), (224, 224))
        patches_savefig(plt, f'vit_patches_{idx}.png', patches, idx_stack, (14, 14), (224, 224),
                        path=input_path)

        exit()

    if 0:  # debug
        for i_batch in range(10):  # first 10 batches
            pixels, attentions = Bs1Atten.load(f'bs1_attn/bs1_attn_{i_batch}.pt')  # ~7.3MB
            Bs1Atten.process(pixels, attentions, i_batch)
        exit()

    ##

    #==== mode -- mnist
    if 0:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        mnist = load_dataset("mnist")
        train_dataset = CustomMnistDataset(mnist["train"], transform=transform)
        val_dataset = None
        test_dataset = CustomMnistDataset(mnist["test"], transform=transform)

        #print('@@ type(train_dataset[0]):', type(train_dataset[0]))  # <class 'tuple'>

        if 0:  # @@ dev; mnist
            #====
            len_train, len_test = 60, 10  # 0.1%; for dev iter
            #====
            #len_train, len_test = 6000, 1000  # 10%
            """ vit_arch_mri_1v1.ipynb
            Epoch 1/8, Average Loss: 1.4045
            Epoch 2/8, Average Loss: 0.8155
            Epoch 3/8, Average Loss: 0.5848
            Epoch 4/8, Average Loss: 0.4377
            Epoch 5/8, Average Loss: 0.3257
            Epoch 6/8, Average Loss: 0.2456
            Epoch 7/8, Average Loss: 0.2265
            Epoch 8/8, Average Loss: 0.1638
            Model saved to custom_vit_mnist.pth
            Test Accuracy: 87.50%
            """
            #====

            train_dataset, _ = random_split(train_dataset, [len_train, len(train_dataset) - len_train])
            test_dataset, _ = random_split(test_dataset, [len_test, len(test_dataset) - len_test])
            print('@@ !! train/test dataset shortened')
        else:
            pass  # 60000, 10000  # 100%
    #==== mode -- mri-erica
    if 0:
        #ds_paths, class_names_sorted = get_mri_ds_paths('debug')
        ds_paths, class_names_sorted = get_mri_ds_paths('erica', root='datasets_mri/50-001')  # colab
        #ds_paths, class_names_sorted = get_mri_ds_paths('erica', root='datasets_mri/50-001-100')  # debug

        stat_ds_paths(ds_paths)

        #cmds = CustomMriDataset(ds_paths['train'], ch=230, rh=160)  # orig
        #cmds = CustomMriDataset(ds_paths['train'], ch=230, rh=80)  # new
        cmds = CustomMriDataset(ds_paths['train'], ch=250, rh=80)  # new, adjusted

        #train_dataset, val_dataset, test_dataset = cmds.random_split([1000, 100, 100])  # colab
        train_dataset, val_dataset, test_dataset, _ = cmds.random_split([80, 10, 10, len(cmds)-100])

        #print(train_dataset[0])  # ok
    #==== mode -- dao
    if 1:
        from ..vit.vit_torch import get_dao_ds_paths
        ds_paths, class_names_sorted = get_dao_ds_paths(  # dao 9-class
            root='datasets_dao', exts=(".png", ".jpg"))  # 7673 samples

        stat_ds_paths(ds_paths)

        cdds = CustomDaoDataset(ds_paths['train'])

        #train_dataset, val_dataset, test_dataset = cdds.random_split([len(cdds)-200, 100, 100])  # colab
        train_dataset, val_dataset, test_dataset, _ = cdds.random_split([80, 10, 10, len(cdds)-100])
    #====

    print(f'train_dataset: {train_dataset.get_histogram(class_names_sorted)}')
    if val_dataset is not None:
        print(f'val_dataset: {val_dataset.get_histogram(class_names_sorted)}')
    print(f'test_dataset: {test_dataset.get_histogram(class_names_sorted)}')

    if 0:
        exit()  # !!

    ##

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = None if val_dataset is None else DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_dataloader_bs1 = DataLoader(test_dataset, batch_size=1, shuffle=False)  # for attention debug

    ##

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    #model = CustomViT(num_classes=10, num_hidden_layers=4).to(device)  # mnist
    #model = CustomViT(num_classes=4, num_hidden_layers=12).to(device)  # erica
    model = CustomViT(num_classes=9, num_hidden_layers=12).to(device)  # dao

    # MODEL_PATH = "custom_vit_mnist.pth"
    #MODEL_PATH = "custom_vit_mnist--10pct-8eps.pth"  # 10% of full size
    #MODEL_PATH = "custom_vit_erica--train90test10-1eps.pth"
    #MODEL_PATH = "custom_vit_erica_colab_8eps.pth"  # full: [1100, 100]
    #MODEL_PATH = "custom_vit_erica_colab_20eps.pth"  # full: [1100, 100], latest (@@ torch.__version__: 2.6.0+cu124)
    MODEL_PATH = "custom_vit_dao--train90test10-1eps.pth"

    #raise Exception("!!!! ok")

    if 0:  # do training?
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        #epoch_n = 8
        epoch_n = 1  # !!!

        save_best_val_model = True
        if save_best_val_model:
            model.custom_train(device, optimizer, loss_fn, epoch_n, train_dataloader,
                               val_dataloader=val_dataloader,
                               save_best_val_model=True,
                               save_path=MODEL_PATH)
        else:
            model.custom_train(device, optimizer, loss_fn, epoch_n, train_dataloader,
                               val_dataloader=val_dataloader,
                               save_best_val_model=False)
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Model saved to {MODEL_PATH}")
    else:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device_str), weights_only=False))
            print(f"Model loaded from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"Error: No saved model found at {MODEL_PATH}. Please train first.")
            exit(1)

    ##

    model.eval()

    y_true = None
    y_pred = None

    if 1:
        print(f"Testing with `test_dataloader`...")
        y_true, y_pred = model.custom_test(device, test_dataloader)
        get_confusion_matrix(y_true, y_pred, class_names_sorted)
        #exit()  # !!

    if 1:
        import os
        from ..vit_finetune.main import verify_attentions

        #attn_dir = 'inference_attention_arch'
        attn_dir = 'inference_attention_arch_dao'

        if not os.path.exists(attn_dir):
            os.makedirs(attn_dir, exist_ok=True)

        # verify_sample_size = 5
        verify_sample_size = 10  # !!!!

        print(f'Verifying first {verify_sample_size} samples of {len(test_dataloader_bs1)}')

        verify_attentions(model, test_dataloader_bs1, verify_sample_size,
                          y_true=y_true, y_pred=y_pred,
                          ckpt_file=MODEL_PATH, save_dir=attn_dir,
                          mri_ch=250, mri_rh=80)

    if 0:  # attention processing debug
        with torch.no_grad():
            for i_batch, batch in enumerate(test_dataloader_bs1):
                pixels, labels, _extra = batch
                pixels, labels = pixels.to(device), labels.to(device)
                #print(f"pixels, labels: {pixels.shape}, {labels.shape}")  # torch.Size([1, 1, 224, 224]), torch.Size([1])

                logits, attentions = model(pixels, output_attentions=True)
                print(f"Number of attention layers: {len(attentions)}")

                if 0:
                    Bs1Atten.save(pixels, labels, logits, attentions, i_batch)  # e.g. 'bs1_attn/*'
                else:
                    Bs1Atten.process(pixels, attentions, i_batch)
                    exit()  # !!

if __name__ == "__main__":
    main()
