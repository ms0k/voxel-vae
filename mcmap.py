import amulet
#import NBT.nbt as nbtlib
import py_vox_io.pyvox.writer
import py_vox_io.pyvox.models
import numpy as np
import tqdm
import random
import mcpalette
import torch

tensor_onehot=False #tensor encoding: one-hot or embedding?

target_map_size=16

mapsave_palette_colors=[[random.getrandbits(8) for c in range(4)] for i in range(256)]
for i in range(len(mcpalette.palette_colors)):
	mapsave_palette_colors[i]=mcpalette.palette_colors[i]

def map3d_load_minecraftmap(path, coord1, coord2):
	if type(path)==str:
		world=amulet.load_level(path)
	else:
		world=path
	blocks=np.zeros([(coord2[0]-coord1[0])*16, 256, (coord2[1]-coord1[1])*16], dtype=np.uint8)+mcpalette.get_unified_palette_ind("minecraft:air")
	dim="minecraft:overworld"
	try:
		for z in tqdm.tqdm(range(coord1[1], coord2[1])):
			for x in range(coord1[0], coord2[0]):
				try:
					chunk=world.get_chunk(x, z, dim)
				except amulet.api.errors.ChunkDoesNotExist:
					continue
				except amulet.api.errors.ChunkLoadError:
					continue
				newpalette=[0]*len(chunk.block_palette)
				for i, p in enumerate(chunk.block_palette):
					newpalette[i]=mcpalette.get_unified_palette_ind(str(p))
				chunkind=np.array([x-coord1[0], z-coord1[1]])*16
				blockref=blocks[chunkind[0]:chunkind[0]+16, :, chunkind[1]:chunkind[1]+16]
				blockref[:, ::-1, :]=chunk.blocks
				blockref[:, :, :]=np.vectorize(lambda x: newpalette[x])(blockref)
	except IOEror:
		pass
	return world, blocks

def map3d_save_as_vox(map, fname):
	map=map.copy()
	map+=1
	map[map==mcpalette.unified_palette["minecraft:air"]+1]=0
	model=py_vox_io.pyvox.models.Vox.from_dense(map)
	model.palette=mapsave_palette_colors
	py_vox_io.pyvox.writer.VoxWriter(fname, model).write()

def map3d_shift_ground(map):
	origysize=map.shape[1]
	air_ind=mcpalette.unified_palette["minecraft:air"]
	for y in range(0, map.shape[1]):
		if np.sum(map[:, y, :]!=air_ind)>(map.shape[0]*map.shape[2])*.995:
			map=map[:, 0:y+1, :]
			break
	return map

def map3d_remove_water(world, map, newind=0):
	for k, v in world.block_palette._block_to_index_map.items():
		if "water" in str(k):
			map[map==v]=0

def map3d_adjust_size(vox_arr, targetsize, air_ind=mcpalette.unified_palette["minecraft:air"]):
	for i in range(3):
		if vox_arr.shape[i]<targetsize:
			uneven=(targetsize-vox_arr.shape[i])%2
			pad_size=(targetsize-vox_arr.shape[i])//2
			shape_arr=list(vox_arr.shape[:])
			shape_arr[i]=pad_size
			newarr=np.zeros(shape_arr, dtype=np.uint8)+air_ind
			vox_arr=np.concatenate((newarr, vox_arr), axis=i)
			if uneven:
				pad_size+=1
				shape_arr[i]=pad_size
				newarr=np.zeros(shape_arr, dtype=np.uint8)+air_ind
			vox_arr=np.concatenate((vox_arr, newarr), axis=i)
	if vox_arr.shape[0]>targetsize:
		pad_size=(vox_arr.shape[0]-targetsize)
		vox_arr=vox_arr[pad_size//2+(pad_size%2):vox_arr.shape[0]-pad_size//2, :, :]
	while vox_arr.shape[1]>targetsize and np.sum(vox_arr[:, 0, :]!=air_ind)==0:
		vox_arr=vox_arr[:, 1:, :]
	if vox_arr.shape[1]>targetsize:
		pad_size=(vox_arr.shape[1]-targetsize)
		vox_arr=vox_arr[:, pad_size//2+(pad_size%2):vox_arr.shape[1]-pad_size//2, :]
	if vox_arr.shape[2]>targetsize:
		pad_size=(vox_arr.shape[2]-targetsize)
		vox_arr=vox_arr[:, :, pad_size//2+(pad_size%2):vox_arr.shape[2]-pad_size//2]
	assert vox_arr.shape[0]==targetsize and vox_arr.shape[1]==targetsize and vox_arr.shape[2]==targetsize
	return vox_arr

tensor_scaling_factor=max(mcpalette.unified_palette.values())

def map3d_from_tensor(vox_arr, from_onehot=tensor_onehot):
	while len(vox_arr.shape)>4:
		vox_arr=vox_arr[0]
	if from_onehot:
		vox_arr=torch.nn.Sigmoid()(vox_arr)
		vox_arr=torch.argmax(vox_arr, dim=0)
	if torch.is_tensor(vox_arr):
		vox_arr=vox_arr.detach().cpu().numpy()
	vox_arr=vox_arr.astype(np.uint8)
	return vox_arr

def map3d_to_tensor(vox_arr, as_onehot=tensor_onehot):
	if len(vox_arr.shape)>3 and vox_arr.shape[0]==1:
		vox_arr=vox_arr[0]
	if not torch.is_tensor(vox_arr):
		vox_arr=torch.tensor(vox_arr)
	vox_arr=vox_arr.to(torch.int64)
	if as_onehot:
		vox_arr=torch.nn.functional.one_hot(vox_arr, num_classes=max(mcpalette.unified_palette.values())+1)
		vox_arr=vox_arr.permute([3, 0, 1, 2]).type(torch.float32)
	return vox_arr

def map3d_apply_palette(map, palette):
	return np.vectorize(lambda x: palette[x])(map)