import torch
from .. import digitake
import os


ARTIFACTS_OUTPUT = './output'

def mk_artifact_dir(dirname):
    path = f'{ARTIFACTS_OUTPUT}/{dirname}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_device():
    digitake.model.set_reproducible(2565)

    use_gpu = os.environ.get('WSDAN_USE_GPU')
    if use_gpu == '1':  # force
        print('@@ get_device(): force GPU settings...')
        assert torch.cuda.is_available(), "Don't forget to turn on gpu runtime!"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    return device

def show_data_loader(data_loader, plt_show=False):
    x = enumerate(data_loader)

    try:
        i, v = next(x)

        shape = v[0].shape
        batch_size = shape[0]
        channel = shape[1]
        w = shape[2]
        h = shape[3]

        print(shape)
        print(f"X contains {batch_size} images with {channel}-channels of size {w}x{h}")
        print(f"y is a {type(v[1]).__name__} of", v[1].tolist())
        print()
        for k in v[2]:
            print(f"{k}=", v[2][k])

    except StopIteration:
        print('StopIteration')

    return channel, batch_size, w, h
