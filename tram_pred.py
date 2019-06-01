
import os
import sys
from collections import deque
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from xgboost import XGBClassifier
from timeblock import TimeBlock
from stackedmodel import StackedModel, make_multiple_categories
from pickle import Pickler, Unpickler
from python_speech_features import mfcc, logfbank

tram_types = ["1_New", "2_CKD_Long", "3_CKD_Short", "4_Old"]
acc_types = ["accelerating", "braking"]
all_types = acc_types + ["negative"]


def create_test_blocks(input_file, window_len=3, time_step=0.2):
	fs, data = wavfile.read(input_file)
	test_blocks = deque()
	step_size = window_len * fs
	for start_time in range(0, len(data), int(time_step * fs)):
		tmp_data = data[start_time:start_time + step_size]
		sound_features = create_sound_features(tmp_data, fs)
		test_blocks.append(sound_features)
	return np.array(test_blocks)


def get_sound_blocks(input_file, mute_model, sound_model, delta=0.6, min_dur=1.5):
	test_sound = create_test_blocks(input_file)
	predict_sound = mute_model.predict(test_sound)
	final_predictions = sound_model.predict(test_sound)
	sound_time = deque()
	last_block = None
	for index, (quiet_ind, final_prediction) in enumerate(zip(predict_sound, final_predictions)):
		if quiet_ind == 0:
			seconds = index * 0.2
			if last_block is None or not last_block.is_within_block(seconds):
				if last_block is not None:
					sound_time.append(last_block)
				last_block = TimeBlock(seconds, final_prediction, delta=delta)
			else:
				last_block.add_new_time(seconds)
				last_block.add(final_prediction)
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


def create_predict_file(input_path, mute_model, tram_model):
	sound_blocks = get_sound_blocks(input_path, mute_model, tram_model, delta=1, min_dur=1.5)
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


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	freqs, times, spec = signal.spectrogram(audio,
	                                        fs=sample_rate,
	                                        window='hann',
	                                        nperseg=nperseg,
	                                        noverlap=noverlap,
	                                        detrend=False)
	return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def create_model_features(source_path):
	X = deque()
	y_type = deque()
	y_acc = deque()

	for root, _, files in os.walk(source_path):
		if any([t_type in root for t_type in tram_types]):
			for file in files:
				tmp_X, tmp_y_type, tmp_y_acc = prepare_file(root, file)
				X.extend(tmp_X)
				y_type.extend(tmp_y_type)
				y_acc.extend(tmp_y_acc)

	X = np.array(X)
	y_type = np.array(y_type).astype(np.uint8)
	y_acc = np.array(y_acc).astype(np.uint8)
	return X, y_type, y_acc


def create_mute_model_features(source_path):
	X_neg = deque()
	y_neg = deque()
	for root, _, files in os.walk(source_path):
		if any([a_type in root for a_type in all_types]):
			for file in files:
				is_mute_file = not any([a_type in root for a_type in acc_types])
				if is_mute_file:
					tmp_X, _, _ = prepare_file(root, file, time_window=3)
				else:
					tmp_X, _, _ = prepare_file(root, file)
				X_neg.extend(tmp_X)
				y_neg.extend(np.ones(len(tmp_X), dtype=np.uint8) * int(is_mute_file))
	return np.array(X_neg), np.array(y_neg)


def create_mute_model(source_path):
	X_neg, y_neg = create_mute_model_features(source_path)
	neg_xgb = XGBClassifier(objective='binary:logistic', eval_metric="auc", gamma=0, learning_rate=0.3, max_depth=3,
	                        min_child_weight=1, n_estimators=300, nthread=-1)
	neg_xgb.fit(X_neg, y_neg)
	return neg_xgb


def create_tram_model(source_path):
	X, y_type, y_acc = create_model_features(source_path)
	model = StackedModel(y_acc, y_type)
	model.fit(X, y_acc, y_type)
	return model


def create_output_csv(target_path, mute_model, tram_model):
	for root, _, files in os.walk(target_path):
		for file in files:
			if file.endswith(".wav"):
				create_predict_file(os.path.join(root, file), mute_model, tram_model)


def main():
	if len(sys.argv) != 3:
		raise Exception("Provide exactly two parameters:[source_folder] [test_folder]")
	source_path = sys.argv[1]
	target_path = sys.argv[2]
	cache_model_path = "final_model.pickle"
	if not os.path.isfile(cache_model_path):
		print("Creating mute model ...")
		mute_model = create_mute_model(source_path)
		print("Creating tram model ...")
		tram_model = create_tram_model(source_path)
		with open(cache_model_path, "wb") as f:
			Pickler(f).dump((mute_model, tram_model))
	else:
		print("Using cached model ...")
		with open(cache_model_path, "rb") as f:
			mute_model, tram_model = Unpickler(f).load()
	print("Predicting files")
	create_output_csv(target_path, mute_model, tram_model)


if __name__ == "__main__":
	main()
