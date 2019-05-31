import operator


class TimeBlock:
	def __init__(self, start, tram_acc_ind, delta=0.6):
		self.delta = delta
		self.start = start
		self.end = start
		self.tram_acc_inds = {}
		self.add(tram_acc_ind)

	def add(self, tram_acc_ind):
		self.tram_acc_inds[tram_acc_ind] = self.tram_acc_inds.get(tram_acc_ind, 0) + 1

	def get_tram_acc_type(self):
		return max(self.tram_acc_inds.items(), key=operator.itemgetter(1))[0]

	def is_within_block(self, time):
		return ((self.start - self.delta) <= time) and (time <= (self.end + self.delta))

	def add_new_time(self, time):
		if self.is_within_block(time):
			self.start = min(self.start, time)
			self.end = max(self.end, time)

	def __repr__(self):
		tab = "\n\t"
		return f"TimeBlock[{tab}start:{self.start},{tab}end:{self.end},{tab}comb:{self.tram_acc_inds}\n]"