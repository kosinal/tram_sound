import os
import sys
from collections import deque
from pickle import Pickler, Unpickler

import math
import numpy as np
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
from xgboost import XGBClassifier

from timeblock import TimeBlock

tram_types = ["1_New", "2_CKD_Long", "3_CKD_Short", "4_Old"]
acc_types = ["accelerating", "braking"]
all_types = acc_types + ["negative"]
threshold = 0.7


def array_contains(input_array, check_array):
	return all([i in check_array for i in input_array])


def convert_to_numpy(input_obj):
	if isinstance(input_obj, np.ndarray):
		return input_obj
	else:
		return np.array(input_obj)


def make_single_number(y_type_input, y_acc_input):
	array_contains(y_acc_input, [0, 1])
	array_contains(y_type_input, [0, 1, 2, 3])

	return convert_to_numpy(y_type_input) + (10 * convert_to_numpy(y_acc_input))


def make_multiple_categories(y_compound):
	y_compound = int(y_compound)
	acc_idx = y_compound // 10
	tram_type_idx = y_compound % 10
	return acc_idx, tram_type_idx


def create_test_blocks(input_file, window_len=3, time_step=0.2):
	fs, data = wavfile.read(input_file)
	test_blocks = deque()
	step_size = window_len * fs
	for start_time in range(0, len(data), int(time_step * fs)):
		tmp_data = data[start_time:start_time + step_size]
		sound_features = create_sound_features(tmp_data, fs)
		test_blocks.append(sound_features)
	return np.array(test_blocks)


def get_sound_blocks(input_file, classifier, delta=0.6, min_dur=1.5):
	test_sound = create_test_blocks(input_file)
	predict_sound = classifier.predict_proba(test_sound)
	sound_time = deque()
	last_block = None
	for index, final_prediction in enumerate(predict_sound):
		argmax_prob = np.argmax(final_prediction)
		max_prob = final_prediction[argmax_prob]
		if max_prob < threshold:
			continue
		predicted_class = int(classifier.classes_[argmax_prob])
		if predicted_class != 0:
			seconds = index * 0.2
			predicted_class -= 1
			if last_block is None or not last_block.is_within_block(seconds):
				if last_block is not None:
					sound_time.append(last_block)
				last_block = TimeBlock(seconds, predicted_class, delta=delta)
			else:
				last_block.add_new_time(seconds)
				last_block.add(predicted_class)
	sound_time.append(last_block)
	return filter_blocks(sound_time, min_dur)


def filter_blocks(sound_time_block, min_dur=1.5):
	return [block for block in sound_time_block if (block.end - block.start) >= min_dur]


def decode_sound_block(block):
	name, decoded = decode_tram(block.get_tram_acc_type())
	return block.start, name, decoded


def decode_tram(index):
	acc_idx, tram_type_idx = make_multiple_categories(index)
	ret_var = np.zeros(len(tram_types) * len(acc_types), dtype=np.uint8)
	ret_var[acc_idx * len(tram_types) + tram_type_idx] = 1
	return f"{acc_types[acc_idx]}_{tram_types[tram_type_idx]}", ret_var


def create_output_line(time, decoded):
	joined_dec = ",".join(str(v) for v in decoded)
	return f"{time:.1f},{joined_dec}"


def create_predict_file(input_path, classifier):
	sound_blocks = get_sound_blocks(input_path, classifier)
	dec_list = [decode_sound_block(block) for block in sound_blocks]
	lines = [create_output_line(time, decoded) for time, _, decoded in dec_list]
	nl = "\n"
	with open(f"{input_path}.csv", "wt") as f:
		f.write(
			f"seconds_offset,accelerating_1_New,accelerating_2_CKD_Long,accelerating_3_CKD_Short,accelerating_4_Old,"
			f"braking_1_New,braking_2_CKD_Long,braking_3_CKD_Short,braking_4_Old{nl}")
		for line in lines:
			f.write(f"{line}{nl}")


def split_into_blocks(data, fs, time_window=3):
	if time_window is None:
		return [data]
	frames_req = int(time_window * fs)
	steps = int(math.ceil(len(data) / frames_req))
	blocks = []
	for i in range(steps):
		blocks.append(data[(i * frames_req):((i + 1) * frames_req)])
	return blocks


def create_sound_features(data, fs):
	bins = 10
	emphasized_data = np.append(data[0], data[1:] - 0.95 * data[:-1])
	result_feature = deque()
	mfcc_feat = mfcc(emphasized_data, fs, nfft=551)
	for space_bin in np.array_split(mfcc_feat, bins):
		result_feature.extend(space_bin.mean(axis=0))
	log_feat = logfbank(emphasized_data, fs, nfft=551)
	for space_bin in np.array_split(log_feat, bins):
		result_feature.extend(space_bin.mean(axis=0))
	return np.array(result_feature)


def prepare_file(root, file, time_window=None):
	tmp_y_type = np.argmax([t_type in root for t_type in tram_types])
	tmp_y_acc = np.argmax([a_type in root for a_type in acc_types])
	fs, data = wavfile.read(os.path.join(root, file))
	blocks = split_into_blocks(data, fs, time_window=time_window)
	tmp_X_datas = []
	for block in blocks:
		sound_features = create_sound_features(block, fs)
		tmp_X_datas.append(sound_features)
	tmp_y_types = np.ones(len(blocks)) * tmp_y_type
	tmp_y_accs = np.ones(len(blocks)) * tmp_y_acc
	return tmp_X_datas, tmp_y_types, tmp_y_accs


def create_one_model_features(source_path):
	X_neg = deque()
	y_neg = deque()
	for root, _, files in os.walk(source_path):
		if any([a_type in root for a_type in all_types]):
			for file in files:
				is_mute_file = not any([a_type in root for a_type in acc_types])
				if is_mute_file:
					tmp_X, _, _ = prepare_file(root, file, time_window=3)
					compound_y = 0
				else:
					tmp_X, tmp_y_type, tmp_y_acc = prepare_file(root, file)
					compound_y = make_single_number([tmp_y_type], [tmp_y_acc])[0] + 1
				X_neg.extend(tmp_X)
				y_neg.extend(np.ones(len(tmp_X), dtype=np.uint8) * compound_y)
	return np.array(X_neg), np.array(y_neg)


def create_model(source_path):
	X, y = create_one_model_features(source_path)
	neg_xgb = XGBClassifier(objective='multi:softmax', eval_metric="auc", gamma=0, learning_rate=0.3,
	                        max_depth=3, min_child_weight=1, n_estimators=300, nthread=-1,
	                        num_class=len(np.unique(y)))
	neg_xgb.fit(X, y)
	return neg_xgb


def create_output_csv(target_path, classifier):
	for root, _, files in os.walk(target_path):
		for file in files:
			if file.endswith(".wav"):
				create_predict_file(os.path.join(root, file), classifier)


def main():
	if len(sys.argv) != 3:
		raise Exception("Provide exactly two parameters:[source_folder] [test_folder]")
	source_path = sys.argv[1]
	target_path = sys.argv[2]
	cache_model_path = "final_model.pickle"
	if not os.path.isfile(cache_model_path):
		print("Creating model ...")
		tram_model = create_model(source_path)
		with open(cache_model_path, "wb") as f:
			Pickler(f).dump(tram_model)
	else:
		print("Using cached model ...")
		with open(cache_model_path, "rb") as f:
			tram_model = Unpickler(f).load()
	print("Predicting files")
	create_output_csv(target_path, tram_model)


if __name__ == "__main__":
	main()
