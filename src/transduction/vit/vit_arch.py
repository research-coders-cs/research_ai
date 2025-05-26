
# pip install torch transformers datasets

##

import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .bs1_atten import Bs1Atten


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

        pretrained_vit = ViTModel.from_pretrained(model_name, output_attentions=True)

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

    def xx_train(self, device, optimizer, loss_fn, epoch_n, train_dataloader):
        for epoch in range(epoch_n):
            epoch_loss = 0.0
            for batch in train_dataloader:
                pixels, labels = batch
                pixels, labels = pixels.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(pixels)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epoch_n}, Average Loss: {epoch_loss / len(train_dataloader):.4f}")

    def xx_test(self, device, test_dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                pixels, labels = batch
                pixels, labels = pixels.to(device), labels.to(device)
                outputs = self(pixels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")


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


class CustomMriDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # c.f. `MriDatasetAdapter` in 'vit_finetune/main.py'
        pixels, class_index, _extra = self.ds[idx]

        return pixels, class_index


def main():

    print('@@ vit arch !!')

    if 0:  # debug
        for i_batch in range(10):  # first 10 batches
            pixels, attentions = Bs1Atten.load(f'bs1_attn/bs1_attn_{i_batch}.pt')  # ~7.3MB
            Bs1Atten.process(pixels, attentions, i_batch)

        exit()

    ##

    #==== mnist
    if 0:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        mnist = load_dataset("mnist")
        train_dataset = CustomMnistDataset(mnist["train"], transform=transform)
        test_dataset = CustomMnistDataset(mnist["test"], transform=transform)
    #==== mri-erica
    if 1:
        processor = CustomViT.get_image_processor()
        print('@@ processor:', processor)

        transf_inner = transforms.Compose([
            transforms.Resize((processor.size['height'], processor.size['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

        from ..vit.vit_torch import stat_ds_paths, get_mri_ds_paths, MriDataset
        #ds_paths, class_names_sorted = get_mri_ds_paths('debug')
        ds_paths, class_names_sorted = get_mri_ds_paths('erica')
        stat_ds_paths(ds_paths)

        transf = lambda pil_img, idx_mri_left_right : transf_inner(
            MriDataset.erica_crop_pil(pil_img, idx_mri_left_right))

        data_set = MriDataset(
            phase='finetune_train',
            dataset=ds_paths['train'],
            transform=transf)
        train_set, test_set, _ = random_split(data_set, [90, 10, len(data_set)-100])

        train_dataset = CustomMriDataset(train_set)
        test_dataset = CustomMriDataset(test_set)

        print('len({train,test}_dataset):', len(train_dataset), len(test_dataset))

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
    # else:
    #     pass  # 60000, 10000  # 100%

    print('@@ len(train_dataset):', len(train_dataset))
    print('@@ len(test_dataset):', len(test_dataset))

    print('@@ type(train_dataset[0]):', type(train_dataset[0]))  # <class 'tuple'>

    ##

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_dataloader_bs1 = DataLoader(test_dataset, batch_size=1, shuffle=False)  # for attention debug

    ##

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    #model = CustomViT(num_classes=10, num_hidden_layers=4).to(device)  # mnist
    model = CustomViT(num_classes=4, num_hidden_layers=12).to(device)  # erica

    # MODEL_PATH = "custom_vit_mnist.pth"
    #MODEL_PATH = "custom_vit_mnist--10pct-8eps.pth"  # 10% of full size
    MODEL_PATH = "custom_vit_erica--train90test10-1eps.pth"

    if 0:  # do training?
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        #epoch_n = 8
        epoch_n = 1  # !!!

        model.train()
        model.xx_train(device, optimizer, loss_fn, epoch_n, train_dataloader)

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    else:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device_str)))
            print(f"Model loaded from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"Error: No saved model found at {MODEL_PATH}. Please train first.")
            exit(1)

    ##

    model.eval()

    if 0:
        print(f"Testing with `test_dataloader`...")
        model.xx_test(device, test_dataloader)
        exit()

    # Optional: Inference with attention debugging
    with torch.no_grad():
        for i_batch, batch in enumerate(test_dataloader_bs1):
            pixels, labels = batch
            pixels, labels = pixels.to(device), labels.to(device)
            #print(f"pixels, labels: {pixels.shape}, {labels.shape}")  # torch.Size([1, 1, 224, 224]), torch.Size([1])

            logits, attentions = model(pixels, output_attentions=True)
            print(f"Number of attention layers: {len(attentions)}")

            if 0:
                Bs1Atten.save(pixels, labels, logits, attentions, i_batch)  # e.g. 'bs1_attn/*'
            else:
                Bs1Atten.process(pixels, attentions, i_batch)


if __name__ == "__main__":
    main()
