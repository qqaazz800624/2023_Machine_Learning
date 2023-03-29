#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import math
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import csv


_exp = 'ensemble'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(77)


class Classifier1(nn.Module):
	def __init__(self, d_model=240, n_spks=600, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(
								d_model=d_model, dim_feedforward=256, nhead=4
									)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.Sigmoid(),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder_layer(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1)
		# mean pooling
		stats = out.mean(dim=1)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out


class Classifier2(nn.Module):
	def __init__(self, d_model=240, n_spks=600, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=6)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model//2),
			nn.ReLU(),
			nn.BatchNorm1d(d_model//2),
			nn.Linear(d_model//2, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder_layer(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1)
		# mean pooling
		stats = out.mean(dim=1)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out

class Classifier3(nn.Module):
	def __init__(self, d_model=240, n_spks=600, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=512, nhead=8)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model//2),
			nn.ReLU(),
			nn.BatchNorm1d(d_model//2),
			nn.Linear(d_model//2, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder_layer(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1)
		# mean pooling
		stats = out.mean(dim=1)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out


class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		return feat_path, mel


def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)

	return feat_paths, torch.stack(mels)


from tqdm.notebook import tqdm


def test_parse_args():
	"""arguments"""
	config = {
		"data_dir": "/neodata/ML/hw4_dataset",
		"model1_path": "/home/u/qqaazz800624/2023_Machine_Learning/HW4/ckpts/d11948002_hw4_model1.ckpt",
		"model2_path": "/home/u/qqaazz800624/2023_Machine_Learning/HW4/ckpts/d11948002_hw4_model2.ckpt",
		"model3_path": "/home/u/qqaazz800624/2023_Machine_Learning/HW4/ckpts/d11948002_hw4_model3.ckpt",
		"output_path": f"/home/u/qqaazz800624/2023_Machine_Learning/HW4/outputs/d11948002_hw4_{_exp}.csv",
		"n_d_model1": 240,
		"n_d_model2": 360,
		"n_d_model3": 512
	}
	return config


def ensemble_main(
	data_dir,
	model1_path,
	model2_path,
	model3_path,
	output_path,
	n_d_model1,
	n_d_model2,
	n_d_model3
	):
	"""Main function."""
	device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())

	dataset = InferenceDataset(data_dir)
	dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    num_workers=8,
                    collate_fn=inference_collate_batch,
	                )
	print(f"[Info]: Finish loading data!",flush = True)

	speaker_num = len(mapping["id2speaker"])
	model1 = Classifier1(d_model=n_d_model1, n_spks=speaker_num).to(device)
	model1.load_state_dict(torch.load(model1_path, map_location=device))
	model1.eval()
	model2 = Classifier2(d_model=n_d_model2, n_spks=speaker_num).to(device)
	model2.load_state_dict(torch.load(model2_path, map_location=device))
	model2.eval()
	model3 = Classifier3(d_model=n_d_model3, n_spks=speaker_num).to(device)
	model3.load_state_dict(torch.load(model3_path, map_location=device))
	model3.eval()

	print(f"[Info]: Finish creating model!",flush = True)

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader):
		with torch.no_grad():
			mels = mels.to(device)
			outputs1 = model1(mels)
			outputs2 = model2(mels)
			outputs3 = model3(mels)
			outs = (outputs1+outputs2+outputs3)/3
			preds = outs.argmax(1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)


if __name__ == "__main__":
	ensemble_main(**test_parse_args())

#%%