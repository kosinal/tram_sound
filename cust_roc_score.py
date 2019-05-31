from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np

def multi_class_scorer(Y_true, Y_pred):
	lb = LabelBinarizer()
	lb.fit(np.concatenate((Y_true, Y_pred)))
	return roc_auc_score(lb.transform(Y_true), lb.transform(Y_pred))

def cust_roc_score():
	return make_scorer(multi_class_scorer)