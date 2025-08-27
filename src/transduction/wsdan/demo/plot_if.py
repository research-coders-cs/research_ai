try:
    import google.colab
    IS_COLAB = True
except:
    IS_COLAB = False

def is_colab():
    return IS_COLAB

#

import matplotlib.pyplot as plt

def get_plt():
    return plt

def plt_show(plt):
    if not is_colab():
        print('@@ plt_show(): \'q\' to close interactively')
    plt.show()
