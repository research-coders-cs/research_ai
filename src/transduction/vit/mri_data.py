import os
import glob
from typing import Dict
import csv


def get_erica_ds_paths(root):
    return EricaDataset.build_ds_paths(root)


class EricaDataset:
    postfix_l = '__l'
    postfix_r = '__r'

    @staticmethod
    def build_ds_paths(root='datasets_mri/50-001'):
        ds_paths = {
            #==== debug <<<<<<<<
            # 'train': EricaDataset.build_datset_erica({
            #     'e1': ['sub-ADNI002S0295_ses-M012__l', 'sub-ADNI002S0295_ses-M036__l',
            #            'sub-ADNI002S0295_ses-M072__r', 'sub-ADNI002S0295_ses-M072__l'],
            #     'e2': ['sub-ADNI002S0295_ses-M012__r', 'sub-ADNI002S0295_ses-M036__r'],
            #     'e3': [],
            #     'e4': [],
            # }, root='datasets_mri/50-001'),  # !!!!
            #==== ok
            'train': EricaDataset.build_datset_erica(
                EricaDataset.load_erica_from_csv(f'{root}/50-001_alisa.csv'),
                root=root),
            #---- !!
            'test': EricaDataset.build_datset_erica({
                # !!!!
            },   root=root),
        }
        return ds_paths

    """
        # head 50-001_alisa.csv
# File,GCA,MTA_RIGHT,MTA_LEFT,ERICA_RIGHT,ERICA_LEFT,PA
# sub-ADNI002S0295_ses-M012,2,1,2,2,1,1 vv
# sub-ADNI002S0295_ses-M036,1,2,2,2,1,1 vv
# sub-ADNI002S0295_ses-M072,2,2,2,1,1,2 vv
# sub-ADNI002S0413_ses-M006,1,2,1,2,1,1 !! <<<<<<<<
# sub-ADNI002S0413_ses-M036,1,2,1,2,1,1
# sub-ADNI002S0413_ses-M060,2,3,2,3,2,2
# sub-ADNI002S0413_ses-M096,2,3,2,3,3,2
# sub-ADNI002S0413_ses-M132,3,3,3,3,3,2
# sub-ADNI002S0559_ses-M012,1,1,1,1,1,2

        # ls datasets_mri/50-001/sub-ADNI002S0295_ses-M*/mta_erica*  # 36 images
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_116.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_118.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_120.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_122.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_124.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_126.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_128.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_130.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_132.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_135.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_138.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M012/mta_erica_sub-ADNI002S0295_ses-M012_140.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_116.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_118.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_120.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_122.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_124.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_126.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_128.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_130.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_132.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_135.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_138.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M036/mta_erica_sub-ADNI002S0295_ses-M036_140.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_116.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_118.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_120.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_122.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_124.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_126.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_128.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_130.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_132.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_135.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_138.png
# datasets_mri/50-001/sub-ADNI002S0295_ses-M072/mta_erica_sub-ADNI002S0295_ses-M072_140.png

        # ls datasets_mri/50-001/sub-ADNI002S0413_ses-M*/mta_erica*  # 60 images
# ...
    """

    @staticmethod
    def load_erica_from_csv(fpath):
        erica_dict = {'E0': [], 'E1': [], 'E2': [], 'E3': [] }

        def erica_append(ed, ex, idx_mri_left_right):
            li = ed.get(f'E{4 - int(ex)}')  # '1' -> 'E3', ..., '4' -> 'E0'
            if li is not None:
                postfix = EricaDataset.postfix_l if idx_mri_left_right else EricaDataset.postfix_r
                li.append(name + postfix)
            else:
                raise ValueError(f'invalid erica score: {ex}')

        with open(fpath, mode='r') as file:
            cr = csv.reader(file)
            for row in cr:
                name = row[0]
                if not name.startswith('sub-'):
                    continue

                erica_append(erica_dict, row[4], 1)  # ERICA_RIGHT <--> MRI_IMAGE_LEFT
                erica_append(erica_dict, row[5], 0)  # ERICA_LEFT <--> MRI_IMAGE_RIGHT

        return erica_dict

    @staticmethod
    def glob_erica(srcdir, root=''):
        if srcdir.endswith(EricaDataset.postfix_l):
            postfix = EricaDataset.postfix_l
            adhoc = '?erica=l'
        elif srcdir.endswith(EricaDataset.postfix_r):
            postfix = EricaDataset.postfix_r
            adhoc = '?erica=r'
        else:
            raise ValueError(f'missing postfix (`{EricaDataset.postfix_l}` or `{EricaDataset.postfix_r}`) for srcdir: {srcdir}')

        return [file + adhoc for file in glob.glob(os.path.join(
            root, srcdir.replace(postfix, ''), 'mta_erica_*.png'))]

    @staticmethod
    def build_datset_erica(datasource: Dict[str, str], root=''):
        datasets = {}
        for key in datasource:
            if isinstance(datasource[key], list):
                files = []
                for path in datasource[key]:
                    #print('!! build_datset_erica(): path:', path)
                    files += EricaDataset.glob_erica(path, root)
                datasets[key] = files
            else:
                datasets[key] = EricaDataset.glob_erica(datasource[key], root)

        return datasets
