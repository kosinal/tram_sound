from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import numpy.testing as npt

class StackedModel:
	def __init__(self, y_acc, y_type):
		y_compound = make_single_number(y_type, y_acc)
		self.typeXG = XGBClassifier(objective='multi:softmax', eval_metric="auc", gamma=0, learning_rate=0.3,
		                            max_depth=3, min_child_weight=1, n_estimators=300, nthread=-1,
		                            num_class=len(np.unique(y_type)))
		self.accXG = XGBClassifier(objective='binary:logistic', eval_metric="auc", gamma=0, learning_rate=0.3,
		                           max_depth=3, min_child_weight=1, n_estimators=300, nthread=-1)
		self.compoundXG = XGBClassifier(objective='multi:softmax', eval_metric="auc", gamma=0, learning_rate=0.3,
		                                max_depth=3, min_child_weight=1, n_estimators=300, nthread=-1,
		                                num_class=len(np.unique(y_compound)))
		self.linearLogReg = LogisticRegression(multi_class="multinomial", C=1.0, penalty="l2", solver="saga")

	def fit(self, X, y_acc, y_type):
		npt.assert_array_equal(np.unique(y_acc), [0, 1])
		npt.assert_array_equal(np.unique(y_type), [0, 1, 2, 3])

		y_compound = make_single_number(y_type, y_acc)
		self.typeXG.fit(X, y_type)
		self.accXG.fit(X, y_acc)
		self.compoundXG.fit(X, y_compound)
		self.linearLogReg.fit(self.__create_stacked_result__(X), y_compound)

	def predict(self, X):
		return self.linearLogReg.predict(self.__create_stacked_result__(X))

	def __create_stacked_result__(self, X):
		return np.hstack((
			self.typeXG.predict_proba(X),
			self.accXG.predict_proba(X),
			self.compoundXG.predict_proba(X))
		)


def make_single_number(y_type_input, y_acc_input):
	return y_type_input + (10 * y_acc_input)


def make_multiple_categories(y_compound):
	acc_idx = y_compound // 10
	tram_type_idx = y_compound % 10
	return acc_idx, tram_type_idx
