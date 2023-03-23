import mcmap
import torch
import numpy as np
import os
import mcschematic
import mcpalette
import gc
import view3dmap
from matplotlib import pyplot as plt

target_nonair_blocks=mcmap.target_map_size*mcmap.target_map_size*mcmap.target_map_size*.00005
ideal_nonair_blocks=mcmap.target_map_size*mcmap.target_map_size*mcmap.target_map_size*.1

def process_3dcraft(schem_dir, target_dir):
	maxsize=[0, 0, 0]
	schemsizes=[[], [], []]
	total_files=0
	total_processed_files=0
	unknowns_hist=[0 for i in range(255)]
	for dir in os.listdir(schem_dir):
		try:
			schem=np.load(schem_dir+"/"+dir+"/schematic.npy")
		except FileNotFoundError:
			continue
		total_files+=1
		air_points=schem[:, :, :, 0]==0
		schem[:, :, :, 0][air_points]=schem[:, :, :, 1][air_points]
		for i in range(3):
			maxsize[i]=max(maxsize[i], schem.shape[i])
		#print(schem.shape)
		size_ok=True
		for s in schem.shape: #ADJUST IF NEEDED
			if s not in range(2, 16):
				size_ok=False
		if not size_ok:
			continue
		for i in range(3):
			schemsizes[i].append(schem.shape[i])
		v=mcmap.map3d_adjust_size(schem[:, :, :, 0], mcmap.target_map_size, air_ind=0)
		old_v=np.copy(v)
		v[:]=np.vectorize(lambda x: mcpalette.craft3dpalette_to_standardpalette[x])(v)
		unknowns=v==mcpalette.unified_palette["etc"]
		for i in range(255):
			unknowns_hist[i]+=np.sum(old_v[unknowns]==i)
		v=v.transpose(1, 2, 0)
		t=torch.tensor(v)[None, :]
		torch.save(t, target_dir+"/"+dir+"_.pt")
		total_processed_files+=1
	f, axes = plt.subplots(3, 1, sharey=True)
	for i in range(3):
		axes[i].hist(schemsizes[i], label="AXIS "+str(i), bins=range(0, 32))
	plt.show()
	print("MAX DIMS", maxsize)
	print("FILES", total_files, ", PROCESSED", total_processed_files)
	print("UNKNOWNS", unknowns_hist)
	for i, h in enumerate(unknowns_hist):
		if h>2000:
			print(i, ":", h)
	print(mcpalette.unified_palette)

process_3dcraft("./rawdata/mcmaps/3dcraft/houses", "./traindata")


print("UNKNOWN:", dict(sorted(mcpalette.palette_unknowns.items(), key=lambda item: item[1])))