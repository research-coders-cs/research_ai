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

from .mri_data import get_erica_ds_paths

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
from torchvision.transforms import ToPILImage, PILToTensor
transform_to_pil = ToPILImage()
transform_to_tensor = PILToTensor()

from PIL import Image
pil_image_open = Image.open

def save_tensor_as_png(fname, ten):
    transform_to_pil(ten).save(fname, format='PNG')

def imread_as_tensor_mri(plt, fpath):
    im = plt.imread(fpath)
    if 0:
        #print('@@ type(im):', type(im))  # <class 'numpy.ndarray'>
        print('@@ im.shape:', im.shape)  # (480, 640, 4)
        print('@@ fpath:', fpath)

    # !! im[:,:,0] == im[:,:,1] == im[:,:,2] (R=G=B), and im[:,:,3] (alpha) is all ones
    if 0:
        print('@@ im[:,:,0].shape:', im[:,:,0].shape)  # (480, 640)
    # print('@@ im[:,:,0]:', im[240:250, 320:330, 0])  # R
    # print('@@ im[:,:,1]:', im[240:250, 320:330, 1])  # G
    # print('@@ im[:,:,2]:', im[240:250, 320:330, 2])  # B
    # print('@@ im[:,:,3]:', im[240:250, 320:330, 3])  # alpha

    if len(im.shape) == 2:
        return torch.tensor([im[:,:]], dtype=torch.float32)  # grayscale
    else:
        return torch.tensor([im[:,:,0]], dtype=torch.float32)  # extract R channel as tensor

# This works also for RGB tensor input (e.g. torch.Size([3, 480, 640]))
def crop_erica_tensor(et):
    ch = 230
    cw = 325
    r = 160
    erica_crop_left =  et[:, ch-r:ch+r, cw-r:cw]  # torch.Size([1, r*2, r])
    erica_crop_right = et[:, ch-r:ch+r, cw:cw+r]  # torch.Size([1, r*2, r])
    #print('@@ erica_crop_left.shape:', erica_crop_left.shape)
    #print('@@ erica_crop_right.shape:', erica_crop_right.shape)

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

class MriDataset(Dataset):

    def __init__(self, phase, dataset, transform):
        assert phase is not None
        assert dataset is not None
        assert transform is not None
        self.phase = phase
        self.dataset = dataset
        self.partition = [(k, len(v)) for k, v in sorted(dataset.items())]
        self.transform = transform

    def __len__(self):
        return len(sum(self.dataset.values(), []))

    def __get_partitioned_index(self, index):
        if index < 0:
            raise IndexError(f"Index must not be negative")

        class_num = 0
        for (k, v) in self.partition:
            if index >= v:
                index -= v
                class_num += 1
            else:
                return k, class_num, index
        raise IndexError(f"Index is out of range {index}")

    def __getitem__(self, index):
        label, class_index, inclass_index = self.__get_partitioned_index(index)
        path = self.dataset[label][inclass_index]
        extra = {
            'path': path,
            'label': label,
            'class_index': class_index,
            'inclass_index': inclass_index
        }
        #print(f'@@ __getitem__(): index: {index} extra: {extra}')

        if path.startswith('datasets_mri00/'):  # !! legacy
            erica_tensor = imread_as_tensor_mri(plt, path)
            erica_crops = crop_erica_tensor(erica_tensor)

            transformed = erica_crops[0]  # left
            ##transformed = erica_crops[1]  # right
            ##transformed = torch.zeros(1, 320, 160)  # erica_crops[0] or erica_crops[1]
        elif self.phase.startswith('finetune_'):
            if path.endswith('?erica=l'):  # !! crude
                transformed = self.transform(
                    pil_image_open(path.replace('?erica=l', '')).convert('RGB'), 0)
            elif path.endswith('?erica=r'):  # !! crude
                transformed = self.transform(
                    pil_image_open(path.replace('?erica=r', '')).convert('RGB'), 1)
            else:
                transformed = self.transform(
                    pil_image_open(path).convert('RGB'))
        else:  # !! legacy
            tens = imread_as_tensor_mri(plt, path)
            transformed = self.transform(tens)
            #print(f'@@ [preprocessing] {tens.shape} -> {transformed.shape}')

        return transformed, class_index, extra

    @staticmethod
    def erica_crop_pil(pil_img, idx_mri_left_right):
        im = transform_to_tensor(pil_img)
        #print(im.dtype, im.shape)  # torch.uint8 torch.Size([3, 480, 640])

        im = crop_erica_tensor(im)  # (left, right)

        if 0:  # debug
            plt_imshow_tensor(plt, im[0])  # left
            plt_imshow_tensor(plt, im[1])  # right

        return transform_to_pil(im[idx_mri_left_right])

    @staticmethod
    def erica_crop_im(im):  # `ndarray` image via `plt.imread()`
        # crop( ndarray
        #       -> torch.Size([480, 640, 4])
        #       -> torch.Size([4, 480, 640])
        #       -> torch.Size([3, 480, 640])  # rgba to rgb
        # )
        crops = crop_erica_tensor(
            torch.from_numpy(im).permute(2, 0, 1)[:3, :, :])
        if 0:  # debug
            plt_imshow_tensor(plt, crops[0])  # left
            plt_imshow_tensor(plt, crops[1])  # right

        return [ crops[idx].permute(1, 2, 0).numpy() for idx in (0, 1) ]


#-------- ^^
import os
import glob
from typing import Dict

def build_dataset(datasource: Dict[str, str], root="", ext="*.png"):
    datasets = {}
    for key in datasource:
        if isinstance(datasource[key], list):
            files = []
            for path in datasource[key]:
                files += glob.glob(os.path.join(root, path, ext))
            datasets[key] = files
        else:
            datasets[key] = glob.glob(os.path.join(root, datasource[key], ext))

    return datasets


def stat_ds_paths(ds_paths):
    print("@@ stat_ds_paths(): ---- ^^")

    for phase, dsp in ds_paths.items():
        if phase in ['train', 'test', 'validate']:
            total = 0
            details = []
            for label in dsp.keys():
                ln = len(dsp[label])
                total += ln
                details.append(f'label={label} {ln}')
            print(f"@@ lens of ds_paths['{phase}']: total: {total} {details}")
        else:
            raise ValueError(f'unknown ds_paths key: {k}')

    print("@@ stat_ds_paths(): ---- $$")
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


#-------- ^^
def _save_ckpt(model, ckpt):
    state_dict = model.state_dict()

    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        #---- TODO
        #'logs': logs,
        #-- e.g. log.txt--mnist-MriViT-MriDataset--ckpt_epochs_5
        # 4229:Epoch 1/5 loss: 2.08
        # 8450:Epoch 2/5 loss: 1.86
        # 12671:Epoch 3/5 loss: 1.78
        # 16892:Epoch 4/5 loss: 1.77
        # 21114:Epoch 5/5 loss: 1.75
        #----
        'state_dict': state_dict}, ckpt)
    print(f'@@ saved: {ckpt}')


def _load_ckpt(model, ckpt):
    ckpt_state_dict = torch.load(ckpt)['state_dict']

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt_state_dict.items()
                       if k in model_dict and model_dict[k].size() == v.size()}

    if len(pretrained_dict) == len(ckpt_state_dict):
        print('%s: ✨ All params loaded' % type(model).__name__)
    else:
        msg = '⚠️ Some params were not loaded'
        print(f'%s: {msg}:' % type(model).__name__)

        not_loaded_keys = [k for k in ckpt_state_dict.keys() if k not in pretrained_dict.keys()]
        print(('  %s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        raise ValueError(msg)

    model_dict.update(pretrained_dict)

    return model_dict
#-------- $$


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

    def save_ckpt(self, ckpt):  # @@
        _save_ckpt(self, ckpt)

    def load_ckpt(self, ckpt):  # @@
        model_dict = _load_ckpt(self, ckpt)
        super(MyViT, self).load_state_dict(model_dict)


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


class MriViT(nn.Module):
    #@@def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
    def __init__(self, chw, n_patches_hw=(8, 4), n_blocks=2, hidden_d=8, n_heads=2, out_d=4):
        # Super constructor
        #@@super(MyViT, self).__init__()
        super(MriViT, self).__init__()  # @@

        # Attributes
        self.chw = chw  # ( C , H , W )
        #@@self.n_patches = n_patches
        self.n_patches_hw = n_patches_hw  # @@
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
            #@@chw[1] % n_patches == 0
            chw[1] % n_patches_hw[0] == 0  # @@
        ), "Input shape not entirely divisible by number of patches"
        assert (
            #@@chw[2] % n_patches == 0
            chw[2] % n_patches_hw[1] == 0  # @@
        ), "Input shape not entirely divisible by number of patches"
        #@@self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
        self.patch_size = (chw[1] / n_patches_hw[0], chw[2] / n_patches_hw[1])  # @@

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            #@@get_positional_embeddings(n_patches**2 + 1, hidden_d),
            get_positional_embeddings(n_patches_hw[0]*n_patches_hw[1] + 1, hidden_d),  # @@
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
        #@@patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        patches = patchify_mri(images, self.n_patches_hw).to(self.positional_embeddings.device)  # @@

        if 0:
            img_hw = (h, w)
            idx = 0  # !!!!
            patches_show(plt, patches, idx, self.n_patches_hw, img_hw)
            patches_savefig(plt, 'patches_idx0.png', patches, idx, self.n_patches_hw, img_hw)
            exit()  # @@ !!!! !!!!

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

    def save_ckpt(self, ckpt):  # @@
        _save_ckpt(self, ckpt)

    def load_ckpt(self, ckpt):  # @@
        model_dict = _load_ckpt(self, ckpt)
        super(MriViT, self).load_state_dict(model_dict)

#-------- ^^ loaders
def get_mnist_ds_paths(debug=False):
    class_dir_map = { f'mnist_{y}': f'y_{y}' for y in range(10) }
    ds_paths = {
        'train': build_dataset(class_dir_map, root='datasets_vit/pngs/train'),
        'test': build_dataset(class_dir_map, root='datasets_vit/pngs/test'),
    }

    if debug:
        stat_ds_paths(ds_paths)

    return ds_paths, sorted(class_dir_map.keys())


def get_thyroid_ds_paths(variant, debug=False):

    if variant == 'ttv':
        ds_paths = {
            'train': build_dataset({
                'benign': ['Train/Benign', 'Val/Benign'],
                'malignant': ['Train/Malignant', 'Val/Malignant'],
            }, root='datasets_thyroid/Dataset_train_test_val'),  # 30 30
            'test': build_dataset({
                'benign': ['Test/Benign'],
                'malignant': ['Test/Malignant'],
            }, root='datasets_thyroid/Dataset_train_test_val'),  # 10 10
        }
    elif variant == '100g':
        ds_paths = {
            'train': build_dataset({
                'benign': ['Markers_Train_Remove_Markers/Benign_Remove/train', 'Markers_Train_Remove_Markers/Benign_Remove/validate'],
                'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/train', 'Markers_Train_Remove_Markers/Malignant_Remove/validate'],
            }, root='Dataset_doppler_100g'),

            'test': build_dataset({
                'benign': ['Markers_Train_Remove_Markers/Benign_Remove/test'],
                'malignant': ['Markers_Train_Remove_Markers/Malignant_Remove/test'],
            }, root='Dataset_doppler_100g'),
            #====
            # 'test': build_dataset({
            #     'benign': ['test26/Benign'],
            #     'malignant': ['test26/Malignant'],
            # }, root='siriraj_original_Testset_26'),
        }
    else:
        raise ValueError(f'unknown ds_paths variant: {variant}')

    if debug:
        stat_ds_paths(ds_paths)

    class_names_sorted = ['benign', 'malignant']

    return ds_paths, class_names_sorted

def get_mri_ds_paths(variant, root='datasets_mri/50-001'):
    if variant == 'debug':
        ds_paths = {
            'train': {  # -> assert len(train_set) == 13
                'e1': ['a', 'b', 'c', 'd'],
                'e2': ['e', 'f', 'g'],
                'e3': ['h', 'i', 'j'],
                'e4': ['k', 'l', 'm'],
            },
            'test': {
                'e1': ['aa', 'bb'],
                'e2': ['p0', 'p1'],
                'e3': ['p0', 'p1'],
                'e4': ['p0', 'p1'],
            },
        }
        class_names_sorted = ['e1', 'e2', 'e3', 'e4']
        ds_paths['train']['e1'][0] = 'datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_116.png?erica=l'
        ds_paths['train']['e1'][1] = 'datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_116.png?erica=r'
    elif variant == 'erica':
        ds_paths = get_erica_ds_paths(root)
        class_names_sorted = ['E0', 'E1', 'E2', 'E3', 'E4']
    else:
        raise ValueError(f'unknown ds_paths variant: {variant}')

    return ds_paths, class_names_sorted

def load_thyroid_data_legacy():
    if 0:  # case thyroid 'Dataset_train_test_val'
        ds_paths, _ = get_thyroid_ds_paths('ttv', debug=False)

        target_resize = (250, 250)  # per `def _train(` in 'src/wsdan/demo/__init__.py'

    if 1:  # case thyroid 100g
        ds_paths, _ = get_thyroid_ds_paths('100g', debug=False)

#        target_resize = (250, 250)  # per `def _train(` in 'src/wsdan/demo/__init__.py'
        target_resize = (280, 280)  # per `def _train(` in 'src/wsdan/demo/__init__.py'

    return create_mri_data(ds_paths, target_resize)

def load_mri_data_legacy():
    ds_paths, _ = get_mri_ds_paths('debug')
    target_resize = (99, 99)  # !!!!

    return create_mri_data(ds_paths, target_resize)


def create_mri_data(ds_paths, target_resize):
    stat_ds_paths(ds_paths)
    print('@@ target_resize:', target_resize)

    train_set = MriDataset(
        phase='train',
        dataset=ds_paths['train'],
        transform=get_transform_mri(target_resize, phase='train'))
    test_set = MriDataset(
        phase='test',
        dataset=ds_paths['test'],
        transform=get_transform_mri(target_resize, phase='test'))

    return train_set, test_set, target_resize


from torchvision import transforms
def get_transform_mri(target_size, phase='train'):
    transform_dict = {
        # 'basic':
        #     transforms.Compose([
        #         transforms.Resize(target_size),
        #         transforms.ToTensor(),
        #         #imagenet_normalize
        #     ]),
        'train':
            transforms.Compose([
                transforms.Resize(target_size),
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(target_size),
            ]),
    }

    if phase in transform_dict:
        return transform_dict[phase]
    else:
        raise Exception("Unknown phase specified")
#-------- $$ loaders

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    # Loading data
    transform = ToTensor()

    if 0:  # case mnist orig
        train_set = MNIST(root="./datasets_vit", train=True, download=True, transform=transform)
        test_set = MNIST(root="./datasets_vit", train=False, download=True, transform=transform)
        #print('@@ type(train_set):', type(train_set))  # <class 'torchvision.datasets.mnist.MNIST'>

        # Defining model and training options
        #==== log.txt--mnist-MyViT
        model = MyViT(
            (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
        ).to(device)

        train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
        n_epochs = 5

    if 1:  # case mnist-MriViT-MriDataset, LGTM
        ds_paths = get_mnist_ds_paths()
        stat_ds_paths(ds_paths)

        train_set = MriDataset(
            phase='train',
            dataset=ds_paths['train'],
            transform=get_transform_mri((28, 28), phase='train'))
        test_set = MriDataset(
            phase='test',
            dataset=ds_paths['test'],
            transform=get_transform_mri((28, 28), phase='test'))

        #==== log.txt--mnist-MriViT-MriDataset
        model = MriViT(  # !!
            (1, 28, 28),  # !!
            n_patches_hw=(7, 7),  # !!
            n_blocks=2, hidden_d=8, n_heads=2,
            out_d=10  # !!
        ).to(device)

        train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
        n_epochs = 5

    if 0:  # case mri
        train_set, test_set, target_resize = load_mri_data_legacy()

        #====
        #train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
        #test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
        #====
        #train_loader = DataLoader(train_set, shuffle=False, batch_size=128)  # *** debug ok
        train_loader = DataLoader(train_set, shuffle=True, batch_size=4)  # debug ok
        train_loader = DataLoader(train_set, shuffle=False, batch_size=4)  # debug ok
        #====

        n_epochs = 5

        model = MriViT(  # !!
            (1, 320, 160),  # !!
            n_patches_hw=(8, 4),  # !!
            n_blocks=2, hidden_d=8, n_heads=2,
            out_d=4  # !!
        ).to(device)

    if 0:  # case thyroid
        train_set, test_set, target_resize = load_thyroid_data_legacy()

        train_loader = DataLoader(train_set, shuffle=True, batch_size=8)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=8)
        n_epochs = 5

        model = MriViT(  # !!
            (1, target_resize[0], target_resize[1]),  # !!
            # n_patches_hw=(25, 25),  # !!
            # n_patches_hw=(10, 10),  # !!
            n_patches_hw=(7, 7),  # !!
            n_blocks=2, hidden_d=8, n_heads=2,
            out_d=2  # !!
        ).to(device)

    #

    if 0:
        train_loop(model, train_loader, device, n_epochs)
        model.save_ckpt(f'vit_patch_NN_resize_MMM_epochs_{n_epochs}.ckpt')
    else:
        #model.load_ckpt('vit_patch_NN_resize_MMM_eps1.ckpt')
        model.load_ckpt('vit_patch_NN_resize_MMM_epochs_5--mnist-MriViT-MriDataset.ckpt')

    test_loop(model, test_loader, device)


def train_loop(model, train_loader, device, n_epochs=5):

    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(n_epochs, desc="Training"):
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
                #continue  # !!!! !!!!

            if 0 and epoch == 0:
                for idx, img in enumerate(x):
                    fname = f'x_ep_{epoch}_batch_{batch_idx}_idx_{idx}_y_{y[idx]}.png'
                    print('@@ saving png for train:', fname)
                    # datasets_vit/pngs (transduction)$ mkdir train/y_{0,1,2,3,4,5,6,7,8,9}
                    save_tensor_as_png(f'./datasets_vit/pngs/train/y_{y[idx]}/{fname}', img)

            if 0:
                plt_imshow_tensor(plt, x[0])
                #plt_imshow_tensor(plt, x[1])

                patches = patchify(x, model.n_patches)
                print('@@ patches.shape:', patches.shape)  # torch.Size([128, 49, 16])
                exit()  # !!!!

            if 0:  # !!!! WIP batch pre-process erica data per '50-001_alisa.csv'
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

                #---- patchify
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

        print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")

        #---- @@
        #if epoch == 0:  # dumps first
        #if epoch == 1:  # dump first and second; NOTE `shuffle=True` for `train_loader`
        if 0 and epoch == n_epochs - 1:  # dump full
            exit()  # !!!!
        #---- @@



def test_loop(model, test_loader, device):

    criterion = CrossEntropyLoss()  # @@

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            #==== ^^
            if 0:
                for idx, img in enumerate(x):
                    fname = f'x_batch_{batch_idx}_idx_{idx}_y_{y[idx]}.png'
                    print('@@ saving png for test:', fname)
                    # datasets_vit/pngs (transduction)$ mkdir test/y_{0,1,2,3,4,5,6,7,8,9}
                    save_tensor_as_png(f'./datasets_vit/pngs/test/y_{y[idx]}/{fname}', img)
            #==== $$
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
