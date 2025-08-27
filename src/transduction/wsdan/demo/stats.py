import torch
import numpy as np

from .plot_if import is_colab, get_plt, plt_show
plt = get_plt()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def print_scores(results):  # @@
    pred = results[2]
    true = results[3]
    _score = 0
    for (i, (y_hat,y)) in enumerate(zip(pred,true)):
        _pred = 'Malignant' if torch.argmax(y_hat) == 1 else "Benign"
        _true = 'Malignant' if y == 1 else 'Benign'
        if _pred == _true:
            _score += 1
        print("Case {}--{} {} Predict:{}---True:{}".format(
            i + 1, softmax(y_hat.cpu().numpy()),
            '✅' if _pred == _true else '❌', _pred, _true))

    print(f'@@ Accuracy: (# of ✅) / (# of Cases) = {_score} / {len(pred)} = %0.3f' % (_score / len(pred)))

def print_auc(results, test_size, plot=False, plot_savepath=None):  # @@
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    # Compute ROC curve and ROC area for each class
    y_pred_b = np.zeros((test_size), dtype=float)
    y_pred_m = np.zeros((test_size), dtype=float)

    pred = results[2]
    true = results[3]

    y_m = true.detach().cpu().numpy()
    y_b = 1 - y_m

    for i, (y_hat, y) in enumerate(zip(pred, true)):
        y_pred_b[i] = float(y_hat[0])
        y_pred_m[i] = float(y_hat[1])

    fpr_b, tpr_b, _ = roc_curve(y_b, y_pred_b)
    fpr_m, tpr_m, _ = roc_curve(y_m, y_pred_m)
    roc_auc_b = auc(fpr_b, tpr_b)
    roc_auc_m = auc(fpr_m, tpr_m)

    print('@@ y_b:', y_b)
    print('@@ y_m:', y_m)
    # print('@@ fpr_b:', fpr_b)
    # print('@@ tpr_b:', tpr_b)
    # print('@@ fpr_m:', fpr_m)
    # print('@@ tpr_m:', tpr_m)
    print('@@ roc_auc_b: %0.3f' % roc_auc_b)
    print('@@ roc_auc_m: %0.3f' % roc_auc_m)

    if plot:
        plt.plot(fpr_b, tpr_b, color = 'darkgreen',
                 lw = 2, label = "ROC Curve for Benign (AUC = %0.3f)" % roc_auc_b)
        plt.plot(fpr_m, tpr_m, color = 'darkred',
                 lw = 2, label = "ROC Curve for Malignant (AUC = %0.3f)" % roc_auc_m)
        plt.plot([0,1.0], [0,1.0], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc = 'lower right')
        if plot_savepath:
            plt.savefig(f'{plot_savepath}/auc.png', bbox_inches='tight')

        plt_show(plt)

def print_poa(results):
    pred = results[2]
    true = results[3]
    for (i, (y_hat, y)) in enumerate(zip(pred, true)):
        perc = softmax(y_hat.cpu().numpy())[1] * 100  # Percentage of Abnormality
        print(f'Case {i + 1} - Abnormality: %0.1f %%' % (perc))
