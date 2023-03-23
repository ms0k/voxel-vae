import torch
import numpy as np

unified_palette_colors={
	"minecraft:dirt": (128, 64, 0, 255),
	"minecraft:cobblestone": (128, 128, 128, 255),
	"minecraft:polished_stone": (96, 96, 96, 255),
	"minecraft:stone": (64, 64, 64, 255),
	"minecraft:water": (0, 0, 64, 128),
	"minecraft:oak_wood": (48, 32, 0, 255),
	"minecraft:oak_leaves": (0, 32, 0, 255),
	"minecraft:air": (0, 0, 0, 0),
	"minecraft:oak_door": (48, 48, 0, 255),
	"minecraft:oak_fence": (42, 42, 0, 255),
	"minecraft:glass": (192, 204, 192, 48),
	"minecraft:bricks": (128, 32, 0, 255),
	#generic decorative material, because buildings can be built from wool and such, but the "etc" material is intended to be a rare special case
	"minecraft:decorative_construction": (128, 0, 123, 123),
	"etc":(255, 0, 255, 255)
}

unified_palette={}
palette_colors=[]
for k,v in unified_palette_colors.items():
	unified_palette[k]=len(unified_palette)
	palette_colors.append(v)

unified_palette_reverse={v: k for k, v in unified_palette.items()}

palette_unknowns={}

air_block_index=unified_palette["minecraft:air"]
craft3d_air_index=0

def get_unified_palette_ind(matstr):
	if matstr.startswith("universal_minecraft"):
		matstr=matstr[len("universal_"):]
	if "[" in matstr:
		matstr=matstr[0:matstr.index("[")]
	if matstr in unified_palette:
		return unified_palette[matstr]
	if matstr.startswith("minecraft:"):
		matstr=matstr[len("minecraft:"):]
	if matstr.startswith("mud_") or "terracotta" in matstr or "sandstone" in matstr or "bricks" in matstr or "quartz" in matstr:
		return unified_palette["minecraft:bricks"]
	if matstr in ["slab", "stairs", "wall", "concrete"] or matstr.endswith("_slab") or matstr.endswith("_pillar"):
		return unified_palette["minecraft:polished_stone"]
	if matstr=="vine":
		return unified_palette["minecraft:oak_leaves"]
	if "_door" in matstr or "door" in matstr:
		return unified_palette["minecraft:oak_door"]
	for leaf in ["leaves", "hay", "plant"]:
		if leaf in matstr:
			return unified_palette["minecraft:oak_leaves"]
	if matstr in ["grass_block", "moss_block"]:
		return unified_palette["minecraft:dirt"]
	if "fence" in matstr:
		return unified_palette["minecraft:oak_fence"]
	if "planks" in matstr or "bark" in matstr:
		return unified_palette["minecraft:oak_wood"]
	for wood_name in ["birch", "oak", "spruce"]:
		if matstr.startswith(wood_name+"_wood"):
			return unified_palette["minecraft:oak_wood"]
	if matstr=="wood" or matstr.endswith("log") or matstr=="chest":
		return unified_palette["minecraft:oak_wood"]
	if matstr.startswith("stripped_") and "wood" in matstr:
		return unified_palette["minecraft:oak_wood"]
	if "cobble" in matstr or "blackstone" in matstr or "sandstone" in matstr:
		return unified_palette["minecraft:cobblestone"]
	if matstr.startswith("minecraft:water") or "snow" in matstr: #melting all the snow :)
		return unified_palette["minecraft:water"]
	if "glass" in matstr:
		return unified_palette["minecraft:glass"]
	if "gravel" in matstr: #TODO: remapping gravel to dirt right now, but planning to have a gravel category
		return unified_palette["minecraft:dirt"]
	if "wool" in matstr or "concrete" in matstr:
		return unified_palette["minecraft:decorative_construction"]
	if "slate" in matstr or "iron_ore" in matstr or "gold_ore" in matstr or "diamond_ore" in matstr or "lapis_ore" in matstr or "redstone_ore" in matstr or "coal_ore" in matstr:
		return unified_palette["minecraft:stone"]
	if matstr not in palette_unknowns:
		palette_unknowns[matstr]=0
	palette_unknowns[matstr]+=1
	return unified_palette["etc"]

def load_craft3d_palette():
	f=open("3dcraft_palette.txt")
	palette_craft3d_to_standard={}
	palette_standard_to_craft3d={}
	start_reading=False
	for l in f.readlines():
		l=l.replace("\n", "")
		if start_reading:
			parts=l.split("	")
			standard_palette_ind=get_unified_palette_ind("minecraft:"+parts[2])
			craft3d_palette_ind=int(parts[0])
			if craft3d_palette_ind not in palette_craft3d_to_standard:
				palette_craft3d_to_standard[craft3d_palette_ind]=standard_palette_ind
				palette_standard_to_craft3d[standard_palette_ind]=craft3d_palette_ind
		start_reading|=not len(l)
	f.close()
	for i in range(255):
		if i not in palette_craft3d_to_standard: #shouldn't really be appearing
			palette_craft3d_to_standard[i]=unified_palette["etc"]
	return palette_craft3d_to_standard, palette_standard_to_craft3d

craft3dpalette_to_standardpalette, standardpalette_to_craft3dpalette=load_craft3d_palette()

def map_summarize_materials(vox_arr):
	if torch.is_tensor(vox_arr):
		vox_arr=vox_arr.detach().cpu().numpy()
	total_blocks=np.sum(vox_arr!=129741365913495741991475981.0)
	for i in range(len(palette_colors)):
		print(unified_palette_reverse[i], np.sum(vox_arr==i)/total_blocks)