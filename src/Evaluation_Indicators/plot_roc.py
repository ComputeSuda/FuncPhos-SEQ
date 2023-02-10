import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import sys 


def load_data(data_file, label_file):
    """
    Load test dataset and test label set
    """

    data = np.load(data_file, allow_pickle=True).astype(float)
    label = np.load(label_file).astype(float).reshape(-1, 1)

    return data, label


def load_models(model_file):
    """
    Load the trained model
    """

    models = []
    for m in model_file:
        model = load_model(m, compile=False)
        models.append(model)

    return models


if __name__ == '__main__':
    data_file = sys.argv[1]
    label_file = sys.argv[2]
    model_file = sys.argv[3]

    data, label = load_data(data_file, label_file)
    models = load_models(model_file)
    
    pred_prob = None
    first = False
    if not first:
        pred_prob = model.predict(data)
        first = True
    else:
        pred_prob += model.predict(data)
    pred_prob /= len(models)
    pred_prob = pred_prob.reshape(-1)
    fpr, tpr, threshold = roc_curve(label, pred_prob)
    roc_auc = str(round(auc(fpr, tpr), 3))
    plt.plot(fpr, tpr, '#4db8cb', label = 'cDL-FuncPhos(AUC=' + roc_auc + ')', linewidth=2)


    # plotting 0.5 line
    plt.legend(loc = 'lower right', fontsize=14)
    plt.plot([0, 1], [0, 1],'r--', linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel('True Positive Rate', size=15, fontdict={'weight':'bold'})
    plt.xlabel('False Positive Rate', size=15, fontdict={'weight':'bold'})
    plt.title('ROC - FuncPhos', fontsize=15, fontdict={'weight':'bold'})
    plt.xticks(fontsize=15, weight='bold')
    plt.yticks(fontsize=15, weight='bold')
    plt.savefig('functional_phosphorylation_auc.png', dpi=500)
    plt.show()  
