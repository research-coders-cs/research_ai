"""
  # local dev setup
  pip install torch==2.2.2
  pip uninstall torchdata torchtext torchvision
  pip install torchdata torchtext torchvision
  pip list | grep torch

  # huggingface setup
  pip install 'transformers[torch]'
  pip install datasets
  pip install scikit-learn
"""

import transduction  # 'research-ai-transduction' pkg (i.e. 'src/transduction/*')

#----
from transduction.plot_if import is_colab, get_plt, plt_show, plt_imshow
plt = get_plt()


def test_plt_show():  # ok
    plt.plot([0,1.0], [0,1.0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if 0 and plot_savepath:
        plt.savefig(f'{plot_savepath}/auc.png', bbox_inches='tight')

    plt_show(plt)

if 0:
    test_plt_show()

def test_df_export():  # ok
    import pandas as pd
    import dataframe_image as dfi
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    dfi.export(df, '__df.png')

if 0:
    test_df_export()


from transduction import example_mask  # !!!!
#----

from transduction.helper import show_example
from transduction.vocab import load_tokenizers, load_vocab  # ok with torch 1.11 (enforced by `torchdata==0.3.0`); err with torch 2.3.0

import logging
logger = logging.getLogger('@@')
logger.setLevel(level=logging.DEBUG if 1 else logging.INFO)

# `mr` when 'src/transduction' is intact
# `mt` when 'src/transduction' is modified

if __name__ == '__main__':

    print("__main__: ^^")

    if 0:
        # global variables used later in the script
        spacy_de, spacy_en = show_example(load_tokenizers)
        vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])

    if 0:  # Encoder/Decoder part, and TODO
        chart = example_mask()  # `alt.Chart`
        fpath = '__chart.png'
        chart.save(fpath)  # require pkgs: `dataframe-image` and `vl-convert-python`
        plt_imshow(plt, fpath)

        # TODO ........


    if 0:  # https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
        #==== mt; slow, needs building the pkg
        #from transduction.vit.vit_torch import main as vit_main
        #==== mr; quick, direct
        from src.transduction.vit.vit_torch import main as vit_main

        vit_main()
        exit()

    if 0:  # finetune
        from src.transduction.vit_finetune.main import main as vit_finetune_main

        vit_finetune_main()
        exit()

    if 1:  # arch
        from src.transduction.vit.vit_arch import main as vit_arch_main

        vit_arch_main()
        exit()
