
# pip install torch transformers datasets

##

import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from ..vit_finetune.attention import get_mask, generate_attention_heatmap##, verify_attentions


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

from transformers import ViTModel, ViTConfig

class CustomViT(nn.Module):
    def __init__(self, num_classes=10, pretrained_model_name="google/vit-base-patch16-224", num_hidden_layers=None):
        super().__init__()

        self.pretrained_config = ViTConfig.from_pretrained(pretrained_model_name)
        if num_hidden_layers is not None:
            self.pretrained_config.num_hidden_layers = num_hidden_layers

        pretrained_vit = ViTModel.from_pretrained(pretrained_model_name, output_attentions=True)

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


class CustomDataset(Dataset):
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

# !!!! ^^
import numpy as np
import cv2
from torchvision.transforms import ToPILImage
transform_to_pil = ToPILImage()
from ..plot_if import get_plt, plt_imshow
plt = get_plt()
# !!!! $$

def process_bs1_attn(pixels, attentions, i_batch, interactive=False):
    for i, attn in enumerate(attentions):
        print(f"Layer {i+1} attention shape: {attn.shape}")  # torch.Size([1, 12, 197, 197])
        # 1 samples, 12 heads (from ViT-Base), 197 tokens (1 CLS + 196 patches).
        """
        The 12 comes from the architecture of the pretrained "google/vit-base-patch16-224" model, which your CustomViT inherits via self.encoder = pretrained_vit.encoder:
        - ViT-Base Specification:
          -- Hidden size: 768.
          -- Number of layers: 12 (using 4 via num_hidden_layers=4).
          -- Number of attention heads: 12 (fixed in the pretrained model).
        - Multi-Head Attention: Each transformer layer in ViT-Base uses multi-head self-attention with 12 heads. Each head processes a portion of the hidden size (768 / 12 = 64 dimensions per head), allowing the model to attend to different aspects of the input simultaneously.
        """
        """
        # Why Not Customizable Here?
        The 12 is hardcoded in the pretrained "google/vit-base-patch16-224" weights. Since you’re leveraging pretrained benefits, the number of heads is fixed to match the loaded weights. If you wanted a different number (e.g., 8), you’d need to:
        - Redefine the attention layers from scratch (losing pretrained weights).
        - Or modify the pretrained layers post-loading (complex and risks breaking compatibility).
        For your MNIST task with pretrained weights, sticking with 12 is optimal and aligns with the original ViT-Base design.

        # What It Means Practically
        - 12 Heads: Each head learns to focus on different patterns in the 197 tokens (CLS + 196 patches). For MNIST, this might mean different heads attend to different parts of the digit (e.g., edges, curves).
        - Debugging: When you inspect attn[0, 0, 0, :5], you’re looking at the first head’s attention from the CLS token to the first 5 patches. The 12 heads give you 12 such perspectives per layer.
        """

        print(f"CLS attention to first 5 patches: {attn[0, 0, 0, :5]}")

    att_mat = torch.cat(attentions).cpu()
    #print(f'@@ att_mat.shape: {att_mat.shape}')  # torch.Size([4, 12, 197, 197]); num_hidden_layers, num_heads, seq_len, seq_len

    # Average the attention weights across all heads.
    ave_att_mat = torch.mean(att_mat, dim=1)
    #print(f'@@ ave_att_mat.shape: {ave_att_mat.shape}')  # torch.Size([4, 197, 197]); num_hidden_layers, seq_len, seq_len

    #print('@@ !! pixels.shape:', pixels.shape)  # torch.Size([1, 1, 224, 224])
    input = pixels.cpu()[0, 0, :, :]  # (224, 224)

    im_mask, joint_attentions, grid_size = get_mask(
        transform_to_pil(input.cpu()), ave_att_mat)
    print('@@ !! im_mask.shape:', im_mask.shape)  # (224, 224)

    im_input = np.stack((input.numpy(),) * 3, axis=-1)  # (224, 224, 3)
    #plt_imshow(plt, im_input)  # debug
    #plt_imshow(plt, im_mask)  # debug

    im_heatmap = transform_to_pil(generate_attention_heatmap(im_input, im_mask))
    if interactive:
        plt_imshow(plt, im_heatmap)
    else:
        fname = f'bs1_attn_heatmap_{i_batch}.png'
        print(f'saving {fname}')
        im_heatmap.save(fname)


def main():

    print('@@ vit arch !!')

    if 1:  # !!!!
        for i_batch in range(10):  # !!!! hardcoded
            debug_attn = torch.load(f'bs1_attn/bs1_attn_{i_batch}.pt')
            pixels = debug_attn['pixels']
            true = debug_attn['true']
            pred = debug_attn['pred']
            attentions = debug_attn['attentions']

            print(f"Predicted: {pred}, True: {true}")
            print(f"Number of attention layers: {len(attentions)}")

            process_bs1_attn(pixels, attentions, i_batch)
        exit()

    ##

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    mnist = load_dataset("mnist")
    train_dataset = CustomDataset(mnist["train"], transform=transform)
    test_dataset = CustomDataset(mnist["test"], transform=transform)

    if 1:  # @@ dev
        #====
        len_train, len_test = 60, 10  # 0.1%; for dev iter
        #====
        #len_train, len_test = 6000, 1000  # 10%
        """ vit_arch_mri_v1.ipynb
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
    model = CustomViT(num_classes=10, num_hidden_layers=4).to(device)

    # MODEL_PATH = "custom_vit_mnist.pth"
    MODEL_PATH = "custom_vit_mnist--10pct-8eps.pth"

    if 0:  # do training?
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        epoch_n = 8

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
            predicted_class = logits.argmax(-1)[0].item()
            print(f"Predicted: {predicted_class}, True: {labels[0].item()}")
            print(f"Number of attention layers: {len(attentions)}")

            if 0:
                print(f'saving bs1_attn_{i_batch}.pt')
                torch.save({'pixels': pixels,
                            'true': labels[0].item(),
                            'pred': predicted_class,
                            'attentions': attentions}, f'bs1_attn_{i_batch}.pt')
            else:
                process_bs1_attn(pixels, attentions, i_batch, interactive=True)


if __name__ == "__main__":
    main()
