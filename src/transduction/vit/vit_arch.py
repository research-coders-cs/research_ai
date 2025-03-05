
# pip install torch transformers datasets

##

import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
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


#-------- ^^
from torch.utils.data import Dataset, random_split

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
#-------- $$


def main():

    print('@@ vit arch !!')

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
        len_train, len_test = 600, 100
# @@ Epoch: 1
# Epoch 1, Loss: 2.0140535831451416
# @@ Epoch: 2
# Epoch 2, Loss: 0.9587066173553467
# @@ Epoch: 3
# Epoch 3, Loss: 0.820209264755249
# Accuracy: 75.0%
        #====
        #len_train, len_test = 6000, 1000
# @@ Epoch: 1
# Epoch 1, Loss: 0.5430163145065308
# @@ Epoch: 2
# Epoch 2, Loss: 0.1401258111000061
# @@ Epoch: 3
# Epoch 3, Loss: 0.48536649346351624
# Accuracy: 88.3%
        #====

        train_dataset, _ = random_split(train_dataset, [len_train, len(train_dataset) - len_train])
        test_dataset, _ = random_split(test_dataset, [len_test, len(test_dataset) - len_test])
        print('@@ !! train/test dataset shortened')
    else:
        pass  # 60000, 10000

    print('@@ len(train_dataset):', len(train_dataset))
    print('@@ len(test_dataset):', len(test_dataset))

    print('@@ type(train_dataset[0]):', type(train_dataset[0]))  # <class 'tuple'>

    ##

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#    test_dataloader = DataLoader(test_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # !!!!!!!!!!

    ##

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomViT(num_classes=10, num_hidden_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    epoch_n = 3
    model.train()
    for epoch in range(epoch_n):
        epoch_loss = 0.0
        for batch in train_dataloader:
            pixels, labels = batch
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixels)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epoch_n}, Average Loss: {epoch_loss / len(train_dataloader):.4f}")

    ##

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            pixels, labels = batch
            pixels, labels = pixels.to(device), labels.to(device)

            logits, attentions = model(pixels, output_attentions=True)
            predicted_class = logits.argmax(-1).item()
            print(f"Predicted: {predicted_class}, True: {labels.item()}")

            # Debug attention
            print(f"Number of attention layers: {len(attentions)}")  # Should be 4
            for i, attn in enumerate(attentions):
                print(f"Layer {i+1} attention shape: {attn.shape}")  # (1, 12, 197, 197)
                print(f"CLS attention to first 5 patches: {attn[0, 0, 0, :5]}")

            break  # !!!!



if __name__ == "__main__":
    main()
