from torch.utils.data import Dataset
from PIL import Image
from transformers.utils.dummy_pt_objects import SquadDataTrainingArguments
from utils import data_utils
from numpy import random, result_type
import os
from collections import defaultdict


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im


class InferenceDatasetWithPath(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.seq_ids = [(path, self.extract_id(path)) for path in self.paths]
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path).convert('RGB').resize((64,64))
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, from_path

	@staticmethod
	def extract_id(path):
		return [int(x[1:]) for x in os.path.splitext(os.path.basename(path))[0].split('_')]


class SeqSampler:

	def __init__(self, seq_ids):
		self.seq_ids = seq_ids

	def __iter__(self):
		res_dict = defaultdict(list)
		for seq_id in sorted(self.seq_ids, key=lambda x: x[1][-1]):
			res_dict['-'.join(seq_id[1][:2])].append(seq_id[0])
		res = [v for k, v in res_dict.items()]
		random.shuffle(res)
		return iter(res)