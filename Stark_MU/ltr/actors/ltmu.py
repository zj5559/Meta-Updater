from . import BaseActor
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
def calculate_acc(pred_label,labels):
    acc = (pred_label == labels).sum().float() / len(labels)
    return acc.item()
def eval(pred_label,labels):
    cnf_matrix = confusion_matrix(labels, pred_label)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return ACC,TPR,TNR
class MUActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective):
        super().__init__(net, objective)

    def __call__(self, data):
        '''
        {'pos_input': pos_input,
                           'neg_input': neg_input,
                           'pos_map': pos_map,
                           'neg_map': neg_map}
        '''
        bs = data['pos_input'].shape[0]
        input = torch.cat((data['pos_input'], data['neg_input']), 0)
        labels1 = np.ones(bs)
        labels2 = np.zeros(bs)
        labels = np.concatenate((labels1, labels2), axis=0)
        labels = torch.tensor(labels, dtype=torch.int64, device=data['pos_input'].device)
        output, _ = self.net(input)
        loss = self.objective(output, labels)
        pred_label = torch.argmax(output, 1)
        acc_all = calculate_acc(pred_label, labels)
        acc, tpr, tnr = eval(pred_label.detach().cpu().numpy(), labels.detach().cpu().numpy())
        stats = {'loss': loss.item(),
                 'acc_all': acc_all,
                 'acc_neg': acc[0],
                 'acc_pos': acc[1],
                 'tpr_neg': tpr[0],
                 'tpr_pos': tpr[1],
                 'tnr_neg': tnr[0],
                 'tnr_pos': tnr[1]
                 }
        return loss, stats

