import math

# %matplotlib inline
import matplotlib.pyplot as plt
import tqdm
from einops import rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
import models.unet
import models.voxception
#import diffusion
import mcpalette
import mcmap
import random
import dataset
import numpy as np
import matplotlib.pyplot as plt
import torchviz
import sys
from torch.utils.data import DataLoader
import os

torch.manual_seed(0)
import mcpalette

batch_size=32 #keeping batch sizes low on purpose, the model works better like this
target_epochs=2000

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"


dataloader = DataLoader(dataset.dataset_all, batch_size=batch_size, shuffle=True)
dataloader_eval = DataLoader(dataset.dataset_all, batch_size=128, shuffle=False)
from pathlib import Path
from torch.optim import Adam

import view3dmap

vq_param={"num_embeddings":16384, "commitment_cost":.25, "decay":.99, "vq":True}
bottleneck="VAE"
model = models.voxception.Voxception_ResNet(in_channels=dataset.dataset_all[0][1].shape[0], base_channels=128, latent_dims=4*0+32-16*int(bottleneck=="VAE"), bottleneck_type=bottleneck, vq_param=vq_param, attention="3d", encoder_depth=3, tensor_format="onehot" if mcmap.tensor_onehot else "embedding")
model.to(device)
optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=.5, total_iters=target_epochs)
bceloss=nn.CrossEntropyLoss(reduction="none") #reduction=sum 82% after 65 epochs on lr 5e-4 1024 latent dims, reduction=none 81%, sum better
air_index=mcpalette.air_block_index


#bceloss=nn.NLLLoss()

def loss_fn(output, target):
	assert(output.shape==target.shape)
	loss=bceloss(output, target)
	air_diff=output[:, air_index, :, :, :]-target[:, air_index, :, :, :] #air where there is supposed to be no air is forbidden
	loss[air_diff>0]*=12
	return torch.sum(loss)

def train(model, epochs, save_every_epoch=10):
	global optimizer, scheduler
	loss_log=[]
	best_loss=10e99
	best_acc=0.0
	best_model=None
	voxel_count=mcmap.target_map_size**3
	start_epoch=0
	model.train()
	model_dec=None
	if len(sys.argv)>1:
		trainstate=torch.load(sys.argv[1])
		scheduler.load_state_dict(trainstate["scheduler"])
		optimizer.load_state_dict(trainstate["optimizer"])
		model=trainstate["model"]
		start_epoch=trainstate["epochs"]+1
		del trainstate
	vae_warmup_epochs=10
	try:
		grad_history={}
		print(model)
		for epoch in tqdm.tqdm(range(start_epoch, epochs)):
			total_loss=0
			total_runs=0
			total_acc=0
			total_perp=0.0
			total_add_loss=0.0
			model.train()
			for step, (batch, labels) in tqdm.tqdm(enumerate(dataloader)): #Embeddings voxels: input = embeddings, output = onehot
				optimizer.zero_grad()
				if not torch.any(batch):
					print("WARNING: EVERYTHING IN TRAINING DATA IS ZERO")
				if torch.any(torch.isnan(batch)):
					raise Exception("TRAINING DATA CONTAINS NAN")
				batch=batch.to(device)
				labels=labels.to(device)
				predicted=model(batch)
				loss=loss_fn(predicted, labels)
				with torch.no_grad():
					total_loss+=torch.sum(loss.detach()).item()
				##https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder #https://arxiv.org/pdf/1511.06349.pdf
				#kl annealing: first training only on reconstruction loss for a few epochs before "switching on" KL
				if model.bottleneck_type in ["VAE", "VQ-VAE"] and epoch>=vae_warmup_epochs:
					#https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
					#"higher values giving more structured latent space at the cost of poorer reconstruction, and lower values giving better reconstruction with less structured latent space"
					#The non-vae autoencoder version reaches great reconstruction accuracies, so we can increase the KL weight
					#For the increased latent space regularity, tradeoff of worse reconstruction is not significant (if observable at all)
					loss+=model.Sampling.kl*20.0
					with torch.no_grad():
						total_add_loss+=model.Sampling.kl.detach().item()
				#loss.backward(torch.ones_like(loss))
				loss.backward()
				optimizer.step()
				with torch.no_grad():
					pred_class=torch.argmax(predicted, dim=1)
					labels_class=torch.argmax(labels, dim=1)
					#print("AIR RATIOS: ", torch.sum(pred_class==air_index)/pred_class.numel(), torch.sum(labels_class==air_index)/labels_class.numel())
					total_runs+=1
					total_acc+=torch.sum(pred_class==labels_class).item()/pred_class.numel()
					if model.Sampling.perplexity is not None:
						total_perp+=model.Sampling.perplexity
			
			with torch.no_grad():
				for name, param in model.named_parameters():
					if param.grad is not None:
						if name not in grad_history:
							grad_history[name]=[]
						grad_history[name].append(param.grad.norm().item())
			optimizer.zero_grad()
			model.eval()
			loss=total_loss/total_runs
			acc=total_acc/total_runs
			if acc>best_acc:
				best_loss=loss
				best_acc=acc
				best_model=model.state_dict()
			print("LOSS: ", loss, "ADD LOSS: ", total_add_loss/total_runs, "LR: ", scheduler.get_last_lr())
			print("ACC: ", acc)
			if model.bottleneck_type=="VQ-VAE":
				print("PERP: ", total_perp/total_runs)
			scheduler.step()
			save_state=(epoch%save_every_epoch)==0 and epoch
			try:
				f=open("save_state.txt", "r")
				save_state=True
				os.system("rm save_state.txt")
			except Exception:
				pass
			if save_state:
				trainstate={"scheduler": scheduler.state_dict(), "optimizer": optimizer.state_dict(), "model": model, "epochs":epoch}
				torch.save(trainstate, "trainstate_%s.pt" % epoch)
				torch.save(grad_history, "grad_history.pt")
			with torch.no_grad():
				total_runs=0
				total_acc=0.0
				for step, (batch, labels) in tqdm.tqdm(enumerate(dataloader_eval)):
					batch=batch.to(device)
					labels=labels.to(device)
					predicted=model(batch)
					pred_class=torch.argmax(predicted, dim=1)
					labels_class=torch.argmax(labels, dim=1)
					total_acc+=torch.sum(pred_class==labels_class).item()/pred_class.numel()
					total_runs+=1
				eval_acc=total_acc/total_runs
				print("SAME DATA EVAL ACC", eval_acc)
			loss_log.append({"loss":loss, "add_loss":total_add_loss/total_runs, "lr":scheduler.get_last_lr(), "accuracy":acc, "perplexity":total_perp/total_runs, "eval_accuracy":eval_acc})
	except KeyboardInterrupt:
		pass
	torch.save(grad_history, "grad_history.pt")
	model.load_state_dict(best_model)
	model.eval()
	return loss_log, best_model


log, best_model=train(model, target_epochs, save_every_epoch=200)
model.load_state_dict(best_model)
torch.save(model, "model.pt")
loss_log=[]
try:
	loss_log=torch.load("loss_log.pt")
except:
	pass
loss_log.append(log)
torch.save(loss_log, "loss_log.pt")
#torch.jit.save(torch.jit.script(model, example_inputs=[next(iter(dataloader))]), "model_scr.pt")