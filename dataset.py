import torch
import os
import mcmap
import mcpalette

class MC3DSnippets(torch.utils.data.Dataset):
	def __init__(self, rootdir):
		self.models=[]
		self.labels=[]
		_vox=[]
		for fname in os.listdir(rootdir):
			if not fname.endswith(".pt"):
				continue
			vox=torch.load(rootdir+"/"+fname)
			if len(_vox)<30:
				_vox.append(vox)
			for i in range(0, 4): #Going through all possible combinations of flipping the model on the X and Y axes
				b=i+1
				axes=[]
				if b&1:
					axes.append(0)
				if b&2:
					axes.append(1)
				#Not flipping houses on their roof because the model ends up remembering that and emitting it...
				#if b&4:
				#	#axes.append(2)
				flipped=torch.flip(vox, axes)
				self.models.append(mcmap.map3d_to_tensor(flipped))
				if not mcmap.tensor_onehot:
					self.labels.append(mcmap.map3d_to_tensor(flipped, as_onehot=True))
			if len(self.models)>=225:
				break
		print("RATIOS OF MATERIALS FOR"+rootdir+":")
		mcpalette.map_summarize_materials(torch.concatenate(_vox))
	def __len__(self):
		return len(self.models)
	def __getitem__(self, idx):
		return self.models[idx], (self.models[idx] if mcmap.tensor_onehot else self.labels[idx])

all_datasets=[
	MC3DSnippets("./traindata"),
]

dataset_all=torch.utils.data.ConcatDataset(all_datasets)