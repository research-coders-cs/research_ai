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

def plt_show(_plt):
    if not is_colab():
        print('@@ plt_show(): \'q\' to close interactively')
    _plt.show()

def plt_imshow_im(_plt, im):
    _plt.figure()
    _plt.imshow(im)
    plt_show(_plt)

def plt_imshow(_plt, x):
    if isinstance(x, str):
        plt_imshow_im(_plt, _plt.imread(x))
    else:
        plt_imshow_im(_plt, x)

def plt_imshow_tensor(_plt, ten, cmap='gray'):
    img = ten.permute(1, 2, 0)  # <c, h, w> -> <h, w, c>
    _plt.imshow(img, cmap=cmap)
    plt_show(_plt)

#

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_confusion_matrix(y_true, y_pred, class_names_sorted):
    cm = get_confusion_matrix_inner(get_plt(), y_true, y_pred, class_names_sorted)
    return cm

def get_confusion_matrix_inner(_plt, y_true, y_pred, class_names_sorted):
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(class_names_sorted))])
    print('@@ cm:\n', cm)
    print(f"@@ accuracy: {100 * accuracy_score(y_pred, y_true):.2f}%")

    fname = 'confusion_matrix.png'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_sorted)

    print(f'@@ get_confusion_matrix(): saving {fname}')
    disp.plot(xticks_rotation=45).figure_.savefig(fname)
    if is_colab():
        plt_imshow(_plt, fname)
        
    return cm
