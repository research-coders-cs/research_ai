from ..vit.vit_torch import stat_ds_paths, random_split_ds_path, build_dataset, \
    get_mnist_ds_paths, get_thyroid_ds_paths, get_mri_ds_paths, MriDataset

#---- ^^
from PIL import Image
from torch.utils.data import Dataset


class ThyroidDataset(Dataset):
    """
    Dataset for Thyroid Image
    """

    def __init__(self, phase, dataset, transform,
                 mask_dict=None, with_alpha_channel=True, ch=None, rh=None):
        """

        :param phase: Train/Validation/Test phase
        :param dataset: the dataset to be loaded(in form on path)
        :param transform: the transform function
        :param mask_dict: (optional) dictionary that map from a given path in the dataset to mask path
        :param with_alpha_channel: (optional) if False, it will load image as RGB(3-channel)
        """
        assert phase is not None
        assert dataset is not None
        assert transform is not None
        self.phase = phase
        self.dataset = dataset
        self.partition = [(k, len(v)) for k, v in sorted(self.dataset.items())]
        self.transform = transform
        self.mask_dict = mask_dict if mask_dict is not None and type(mask_dict) == dict else {}
        self.extra_channel_default = None
        self.with_alpha_channel = with_alpha_channel
        self.ch = ch
        self.rh = rh

        print('@@ ThyroidDataset.__init__(): phase, ch, rh:', phase, ch, rh)

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.partition = [(k, len(v)) for k, v in sorted(self.dataset.items())]  # Create a partition indices

    def __len__(self):
        size = len(sum(self.dataset.values(), []))  # monoid flatten, it's counting item so order doesn't matter
        return size

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
        """
        __getitem__ takes index of linear data, meaning that the label key will be used to keep track of partition
        :param index: linear index
        :return: image, label, extra
        """
        # convert linear index into index respecting its partition
        # e.g. [0,1,2,3,4,5,6,7,8] --> [0,1,2,3,0,1,2,3,4]
        label, class_index, index = self.__get_partitioned_index(index)
        path = self.dataset[label][index]

        extra = {
            'path': path,
            'label': label,
            'class_index': class_index,
            'inclass_index': index
        }

        try:
            extracted_filename = path.split('/')[-1]  # extract the filename of image to find its counterpart
            mask_path = next(p for p in self.mask_dict[label] if extracted_filename in p)
        except StopIteration:
            mask_path = None
        except KeyError:
            mask_path = None

        if self.with_alpha_channel:
            # if it has mask, find the mask path pair and load
            if mask_path:
                # Gray scale image(this could actually be just B/W Image(0/1)
                mask_image = Image.open(mask_path).convert('L')
                r, g, b = image.split()
                image = Image.merge('RGBA', (r, g, b, mask_image))
            else:
                gray_image = Image.open(path).convert('L')
                if self.extra_channel_default and type(self.extra_channel_default) == int:
                    gray_image.point(lambda _i: self.extra_channel_default)
                r, g, b = image.split()
                image = Image.merge('RGBA', (r, g, b, gray_image))
        else:
            if path.endswith('?erica=l'):
                image = MriDataset.erica_crop_pil(Image.open(path.replace('?erica=l', '')).convert('RGB'), 0, ch=self.ch, rh=self.rh)
            elif path.endswith('?erica=r'):
                image = MriDataset.erica_crop_pil(Image.open(path.replace('?erica=r', '')).convert('RGB'), 1, ch=self.ch, rh=self.rh)
            else:
                image = Image.open(path).convert('RGB')

        transformed_image = self.transform(image)

        # return image and label
        return transformed_image, class_index, extra

    def get_class_label(self, class_index):
        assert class_index < len(self.partition), 'The class_index is beyond number of class available'
        return self.partition[class_index][0]
#---- $$
