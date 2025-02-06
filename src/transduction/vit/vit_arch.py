"""
Let's create a minimal example of customizing a Hugging Face transformer layer architecture
for the MNIST dataset. We will create a custom transformer-based model
for classifying handwritten digits from the MNIST dataset.
"""

# pip install torch transformers datasets

##

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class CustomConfig(PretrainedConfig):
    model_type = "custom_model"

    def __init__(self, hidden_size=256, num_hidden_layers=4, num_attention_heads=8, num_classes=10,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes


"""
The Hugging Face `PreTrainedModel` is compatible with the "google/vit-base-patch16-224" model.
This model is a Vision Transformer (ViT) pre-trained on ImageNet-21k and fine-tuned
on ImageNet 2012 - https://huggingface.co/google/vit-base-patch16-224
"""
class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Linear(28 * 28, config.hidden_size)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads) for _ in range(config.num_hidden_layers)])
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        self.init_weights()

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.embeddings(pixel_values.view(batch_size, -1))
        hidden_states = embeddings.unsqueeze(1)  # Add sequence dimension

        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        logits = self.classifier(hidden_states.squeeze(1))  # Remove sequence dimension

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

    AutoConfig.register("custom_model", CustomConfig)
    AutoModel.register(CustomConfig, CustomModel)

    ##

    transform = transforms.Compose([
        #---- TODO
        #Resize((processor.size['height'], processor.size['width'])),
        # TODO organize the sizes
        #----
        transforms.ToTensor(),
        #---- cf.
        # Normalize(mean=processor.image_mean, std=processor.image_std),
        #----
    ])
    preprocess = lambda pil_img : transform(pil_img)

    mnist = load_dataset("mnist")
    train_dataset = CustomDataset(mnist["train"], transform=preprocess)
    test_dataset = CustomDataset(mnist["test"], transform=preprocess)

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
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    ##

    model = CustomModel(CustomConfig())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # Train for 3 epochs
        print(f'@@ Epoch: {epoch+1}')
        for batch in train_dataloader:
            pixels, labels = batch

            optimizer.zero_grad()
            outputs = model(pixels)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    ##

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            pixels, labels = batch

            outputs = model(pixels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    ##


if __name__ == "__main__":
    main()
