import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
from tqdm.notebook import tqdm as tqdm_xx, trange as trange_xx  # @@

np.random.seed(0)
torch.manual_seed(0)

from ..plot_if import get_plt, plt_imshow, plt_imshow_tensor  # @@
plt = get_plt()

def patchify(images, n_patches):
    print('@@ patchify(): ^^ images.shape:', images.shape)
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2)
    patch_size = h // n_patches
    print('@@ patchify(): patches.shape:', patches.shape)
    print('@@ patchify(): patch_size:', patch_size)

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                if 0 and idx == 0 and i == 3:  # @@ for j in range(7)
                    print(f'idx={idx} i={i} j={j} patch.shape={patch.shape} patch:', patch)  # ... patch.shape=torch.Size([1, 4, 4]) ...
                    plt_imshow_tensor(plt, patch)  # @@
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def patchify_mri(images, n_patches_hw):  # @@
    #print('@@ patchify_mri(): ^^ images.shape:', images.shape)  # e.g. torch.Size([1, 1, 320, 160])

    n, c, h, w = images.shape

    n_patches_h, n_patches_w = n_patches_hw
    patches = torch.zeros(n, n_patches_h * n_patches_w, h * w * c // (n_patches_h * n_patches_w))
    patch_size_h = h // n_patches_h
    patch_size_w = w // n_patches_w
    if 0:
        print('@@ patchify_mri(): patches.shape:', patches.shape)
        print('@@ patchify_mri(): patch_size_h:', patch_size_h)
        print('@@ patchify_mri(): patch_size_w:', patch_size_w)

    for idx, image in enumerate(images):
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[
                    :,
                    i * patch_size_h : (i + 1) * patch_size_h,
                    j * patch_size_w : (j + 1) * patch_size_w,
                ]
                patches[idx, i * n_patches_w + j] = patch.flatten()
    return patches


#-------- ^^ @@
def imread_as_tensor_mri(plt, fpath):
    im = plt.imread(fpath)
    print('@@ type(im):', type(im))  # <class 'numpy.ndarray'>
    print('@@ im.shape:', im.shape)  # (480, 640, 4)

    # !! im[:,:,0] == im[:,:,1] == im[:,:,2] (R=G=B), and im[:,:,3] (alpha) is all ones
    print('@@ im[:,:,0].shape:', im[:,:,0].shape)  # (480, 640)
    # print('@@ im[:,:,0]:', im[240:250, 320:330, 0])  # R
    # print('@@ im[:,:,1]:', im[240:250, 320:330, 1])  # G
    # print('@@ im[:,:,2]:', im[240:250, 320:330, 2])  # B
    # print('@@ im[:,:,3]:', im[240:250, 320:330, 3])  # alpha

    return torch.tensor([im[:,:,0]], dtype=torch.float32)  # extract R channel as tensor

def crop_erica_tensor(et):
    ch = 230
    cw = 325
    r = 160
    erica_crop_left =  et[:, ch-r:ch+r, cw-r:cw]  # torch.Size([1, r*2, r])
    erica_crop_right = et[:, ch-r:ch+r, cw:cw+r]  # torch.Size([1, r*2, r])
    print('@@ erica_crop_left.shape:', erica_crop_left.shape)
    print('@@ erica_crop_right.shape:', erica_crop_right.shape)

    return erica_crop_left, erica_crop_right

#-------- $$ @@

#-------- ^^ @@
def patches_show(plt, patches, idx, n_patches_hw, img_hw):
    patches_plot(plt, patches, idx, n_patches_hw, img_hw)
    plt.show()

def patches_savefig(plt, fpath, patches, idx, n_patches_hw, img_hw):
    patches_plot(plt, patches, idx, n_patches_hw, img_hw)
    plt.savefig(fpath, bbox_inches='tight')

def patches_plot(plt, patches, idx, n_patches_hw, img_hw):
    fig = plt.figure()
    rows, cols = n_patches_hw
    patch_h = int(img_hw[0] / n_patches_hw[0])
    patch_w = int(img_hw[1] / n_patches_hw[1])

    axes = []
    for patch in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, patch + 1))

        img = patches[idx, patch]
        img = torch.stack([torch.reshape(img, (patch_h, patch_w))], dim=0)
        plt.imshow(img.permute(1, 2, 0), cmap='gray')

    fig.suptitle(f'# of patches: {rows * cols} (={rows}x{cols})\n'
                 f'patch size: {patch_h * patch_w}(={patch_h}x{patch_w})')

    plt.axis('off')
    plt.setp(axes, xticks=[], yticks=[])  # https://stackoverflow.com/questions/25124143/get-rid-of-tick-labels-for-all-subplots/25127092#25127092
#-------- $$

#-------- ^^ @@
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

class MriDataset(Dataset):

    def __init__(self, phase, dataset, transform):
        assert phase is not None
        assert dataset is not None
        assert transform is not None
        self.phase = phase
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(sum(self.dataset.values(), []))

    def __getitem__(self, index) -> T_co:

        # label, class_index, index = self.__get_partitioned_index(index)
        # path = self.dataset[label][index]
        #
        extra = {
            # 'path': path,
            # 'label': label,
            # 'class_index': class_index,
            # 'inclass_index': index
        }
        #==== @@
        class_index = index + 1000  # !!!! WIP

        print(f'@@ __getitem__(): index: {index} class_index: {class_index}')

        #transformed_image = self.transform(image)
        #==== !!!! WIP
        image = None
        transformed_image = torch.zeros(1, 320, 160)  # erica_crops[0] or erica_crops[1]

        # return image and label
        #return transformed_image, class_index, extra
        return transformed_image, class_index  # @@


def get_transform_mri(target_size, phase='train'):
    transform_dict = {
        'basic':
            transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                #imagenet_normalize
            ]),
        'train':
            transforms.Compose([
                # !!!!
            ]),
        'test':
            transforms.Compose([
                # !!!!
            ]),
    }

    if phase in transform_dict:
        return transform_dict[phase]
    else:
        raise Exception("Unknown phase specified")


def load_data_mri():
    # './datasets_mri/50-001',
    ds_paths = {
        'train': {  # -> assert len(train_set) == 13
            '1': ['a', 'b', 'c', 'd'],
            '2': ['e', 'f', 'g'],
            '3': ['h', 'i', 'j'],
            '4': ['k', 'l', 'm'],
        },
        'test': {
            '1': ['aa', 'bb'],
            '2': ['p0', 'p1'],
            '3': ['p0', 'p1'],
            '4': ['p0', 'p1'],
        },
    }

    # dump `ds_paths` stat
    for phase, dsp in ds_paths.items():
        if phase in ['train', 'test']:
            total = 0
            details = []
            for cls in dsp.keys():
                ln = len(dsp[cls])
                total += ln
                details.append(f'cls={cls} {ln}')
            print(f"@@ lens of ds_paths['{phase}']: total: {total} {details}")
        else:
            raise ValueError(f'unknown ds_paths key: {k}')

    target_resize = (99, 99)  # dummy
    train_set = MriDataset(
        phase='train',
        dataset=ds_paths['train'],
        transform=get_transform_mri(target_resize, phase='train'))
    test_set = MriDataset(
        phase='test',
        dataset=ds_paths['test'],
        transform=get_transform_mri(target_resize, phase='test'))

    return train_set, test_set

#-------- $$


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLPk
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)  # Map to output dimension, output category distribution


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


def main():
    # Loading data
    transform = ToTensor()

    if 1:  # @@
        train_set, test_set = load_data_mri()

        #train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
        #train_loader = DataLoader(train_set, shuffle=False, batch_size=128)  # debug ok
        train_loader = DataLoader(train_set, shuffle=True, batch_size=4)  # debug ok

        test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
    else:  # orig
        train_set = MNIST(
            root="./datasets_vit", train=True, download=True, transform=transform
        )
        test_set = MNIST(
            root="./datasets_vit", train=False, download=True, transform=transform
        )
        #print('@@ type(train_set):', type(train_set))  # <class 'torchvision.datasets.mnist.MNIST'>

        train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )
    model = MyViT(
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)
    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
#====
#@@     for batch in tqdm(
#@@         train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
#@@     ):
#==== @@
        for batch_idx, batch in enumerate(tqdm_xx(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        )):
            print('@@ batch_idx:', batch_idx)  # e.g. MNIST -> 0:469
#====
            x, y = batch
            x, y = x.to(device), y.to(device)
            #==== ^^
            if 1:
                # print('@@ type(batch):', type(batch))  # <class 'list'>
                # print('@@ len(batch):', len(batch))  # 2
                # print('@@ type(x):', type(x))  # <class 'torch.Tensor'>
                # print('@@ type(y):', type(y))  # <class 'torch.Tensor'>
                print('@@ x.shape:', x.shape)  # torch.Size([128, 1, 28, 28])  (n: batch_size, c, h, w)
                print('@@ y.shape:', y.shape)  # torch.Size([128])
                print('@@ y:', y)

                #torch.save(x, 'x_aka_images.pt')  # !!!!
                #xx = torch.load('x_aka_images.pt')
                #print('@@ xx.shape:', xx.shape)  # ok

                #exit()  # !!!! !!!!
                continue  # !!!! !!!!

            if 0:
                plt_imshow_tensor(plt, x[0])
                #plt_imshow_tensor(plt, x[1])

                patches = patchify(x, model.n_patches)
                print('@@ patches.shape:', patches.shape)  # torch.Size([128, 49, 16])
                exit()  # @@ !!!! !!!!

            if 1:  # !!!! WIP batch pre-process erica data per '50-001_alisa.csv'
                fpath = 'datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_116.png'
                #plt_imshow(plt, fpath)

                #---- read & crop
                erica_tensor = imread_as_tensor_mri(plt, fpath)

                print('@@ erica_tensor.shape:', erica_tensor.shape)  # torch.Size([1, 480, 640])
                #plt_imshow_tensor(plt, erica_tensor)

                erica_crops = crop_erica_tensor(erica_tensor)
                _c, crop_h, crop_w = erica_crops[0].shape
                #print(crop_h, crop_w)
                #plt_imshow_tensor(plt, erica_crops[0])  # left
                #plt_imshow_tensor(plt, erica_crops[1])  # right

                #---- patchfy
                n_patches_hw = (8, 4)

                images_mri = torch.stack([erica_crops[0]], dim=0)  # -> torch.Size([1, 1, 320, 160])
                #images_mri = torch.stack([erica_crops[1]], dim=0)  # -> torch.Size([1, 1, 320, 160])
                ##images_mri = torch.stack([erica_crops[0], erica_crops[1]], dim=0)  # -> torch.Size([2, 1, 320, 160])

                patches = patchify_mri(images_mri, n_patches_hw)
                print('@@ patches.shape:', patches.shape)  # torch.Size([1, 32, 1600])

                if 1:
                    img_hw = (crop_h, crop_w)
                    idx = 0  # !!!!
                    patches_show(plt, patches, idx, n_patches_hw, img_hw)
                    patches_savefig(plt, 'patches_idx0.png', patches, idx, n_patches_hw, img_hw)

                exit()  # @@ !!!! !!!!

            #==== $$
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        #---- @@ !!!!
        if epoch == 0:  # !!!! dumps first
        #if epoch == 1:  # !!!! dump first and second; NOTE `shuffle=True` for `train_loader`
            exit()
        #---- @@

    # @@ TODO save weights !!!!!!!!

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
