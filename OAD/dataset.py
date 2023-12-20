import torch
import random
import json
import numpy as np
import torch.utils.data as data
import os

from torch.utils.data import DataLoader
from config import *

class THUMOS14_Train(data.Dataset):
	def __init__(self, class_agno=True):
		with open(ANNOTATION_PATH, 'r', encoding='utf-8') as fp:
			self.meta = json.load(fp)
		self.subset = 'train'								# train									
		self.class_agno = class_agno						# class agnostic or not
		self.slice_idxes, self.labels = self.get_slice_idxes_and_labels(class_agno)
		# slice each video features to fixed length (64)

		print("="*40)
		print("THUMOS14 Dataset Init Complete.")
		print("class agnoistic: ", self.class_agno)
		print("Total number of video : ", len(self.labels))
		print("Total number of feature chunk : ", len(self.slice_idxes))
		print("="*40,'\n')

	def __len__(self):
		return len(self.slice_idxes)

	def __getitem__(self, index):
		vidname, start_idx, end_idx = self.slice_idxes[index]	
		feature = torch.from_numpy(np.load(f'{FEATURE_PATH}{vidname}.npy'))
		# feature : RGB+flow feature
		label = self.labels[vidname]
		ret = {}
		ret['vidname'] = vidname
		ret['data'] = feature[start_idx:end_idx]
		ret['label'] = label[start_idx:end_idx]
		if len(ret['data']) < FEATURE_CHUNK_LEN:
			pad_len = FEATURE_CHUNK_LEN - len(ret['data'])
			feature_padding = torch.zeros(pad_len, FEATURE_SIZE)
			if not self.class_agno:
				label_padding = torch.zeros(pad_len, NUM_CLASSES, dtype=torch.int64)
			else:
				label_padding = torch.zeros(pad_len, 2, dtype=torch.int64)
			ret['data'] = torch.cat((ret['data'], feature_padding), 0)
			ret['label'] = torch.cat((ret['label'], label_padding), 0)
		assert len(ret['data']) == len(ret['label'])
		return ret

	def get_slice_idxes_and_labels(self, class_agno):
		slice_idxes = []
		# [ [file_name, feature_start_idx, feature_end_idx] ... ]
		labels = {}
		# {file_name:label_seq ... }
		for vidname in self.meta['database']:
			if(self.meta['database'][vidname]['subset'] == self.subset):
				feature_len = len(np.load(f'{FEATURE_PATH}{vidname}.npy'))
				label = np.load(f'{LABEL_PATH}{vidname}.npy')
				assert feature_len == len(label)
				if class_agno:
					tmp = np.array([1,0], dtype=np.float32)
					tmp = np.tile(tmp, (feature_len, 1))
					index = np.logical_not(np.logical_or(label[:,0]==1, label[:,-1]==1))
					tmp[index, 0] = 0
					tmp[index, 1] = 1
					label = torch.from_numpy(tmp)
				else:
					label = torch.from_numpy(label)
				labels[vidname] = label
				# if video is too short, just use full feature
				if feature_len < FEATURE_CHUNK_LEN:
					slice_idxes.append([vidname, 0, feature_len])
				else:
					# slice the feature to a fixed length
					start_idx = random.randrange(0, min(FEATURE_CHUNK_LEN, feature_len - FEATURE_CHUNK_LEN + 1))
					end_idx = start_idx + FEATURE_CHUNK_LEN
					while(end_idx <= feature_len):
						slice_idxes.append([vidname, start_idx, end_idx])
						start_idx += FEATURE_CHUNK_LEN
						end_idx += FEATURE_CHUNK_LEN
		return slice_idxes, labels


class THUMOS14_Test(data.Dataset):
	def __init__(self, class_agno=True):
		with open(ANNOTATION_PATH, 'r', encoding='utf-8') as fp:
			self.meta = json.load(fp)
		self.subset = 'test'									# train									
		self.class_agno = class_agno							# test								
		self.video_idxes, self.labels = self.get_video_idxes_and_labels(class_agno)
		print("="*40)
		print("THUMOS14 Test Dataset Init Complete.")
		print("class agnoistic: ", self.class_agno)
		print("Total number of video : ", len(self.labels))
		print("="*40,'\n')

	def __len__(self):
		return len(self.video_idxes)

	def __getitem__(self, index):
		vidname = self.video_idxes[index]
		feature = np.load(f'{FEATURE_PATH}{vidname}.npy')
		# feature : RGB+flow feature
		label = self.labels[vidname]
		ret = {}
		ret['vidname'] = vidname
		ret['data'] = feature
		ret['label'] = label
		assert len(ret['data']) == len(ret['label'])
		return ret

	def get_video_idxes_and_labels(self, class_agno):
		video_idxes = []
		# [file_name ... ]
		labels = {}
		# {file_name:label_seq ... }
		for vidname in self.meta['database']:
			if vidname in SUBTRACT_LIST:
				continue
			if(self.meta['database'][vidname]['subset'] == self.subset):
				feature_len = len(np.load(f'{FEATURE_PATH}{vidname}.npy'))
				label = np.load(f'{LABEL_PATH}{vidname}.npy')
				assert feature_len == len(label)
				if class_agno:
					tmp = np.array([1,0], dtype=np.float32)
					tmp = np.tile(tmp, (feature_len, 1))
					index = np.logical_not(np.logical_or(label[:,0]==1, label[:,-1]==1))
					tmp[index, 0] = 0
					tmp[index, 1] = 1
					label = torch.from_numpy(tmp)
				else:
					label = torch.from_numpy(label)
				labels[vidname] = label
				#video idx
				video_idxes.append(vidname)
		return video_idxes, labels

if __name__ == '__main__':
	thumos14_dataset = THUMOS14_Test(class_agno=True)
	for item in DataLoader(thumos14_dataset, batch_size=1, shuffle=True):
		print("video name: ", item['vidname'])
		print("feature shape : ", item['data'].shape)
		print("label seq shape : ", item['label'].shape)