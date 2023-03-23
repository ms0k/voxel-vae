#https://arxiv.org/pdf/1608.04236.pdf
#"Generative and Discriminative Voxel Modeling with Convolutional Neural Networks" by Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston
#Model inspired by Voxception described in the paper
import torch
from torch import nn
import models.sca_3d
import models.vq_vae


class Passthrough(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
	def forward(self, input):
		return input

class Norm3d(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()
		#self.norm=nn.InstanceNorm3d(*args, **kwargs)
		self.norm=nn.BatchNorm3d(*args, **kwargs)
	def forward(self, input):
		#return self.norm(input) if input.shape[2]>1 else input
		return self.norm(input)

def NormalizationLayer3d(channels):
	return nn.BatchNorm3d(channels)
def NormalizationLayer1d(channels):
	return nn.BatchNorm1d(channels)

class Voxception_Downsample(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.batchnorm=NormalizationLayer3d(in_channels//2)
		self.conv3=nn.Conv3d(in_channels, in_channels//2, 3, padding="same")
		#We have standard strided convolution as well, because pooling destroys position information
		#https://imaging-in-paris.github.io/seminar/slides/newson.pdf
		self.conv3s2=nn.Conv3d(in_channels, in_channels//2, 3, stride=2, padding=1)
		self.conv1s2=nn.Conv3d(in_channels, in_channels//2, 1, stride=2)
		self.map=nn.MaxPool3d(2)
		self.avp=nn.AvgPool3d(2)
		self.relu=nn.ELU()
		self.sca3d=models.sca_3d.SCA3D(channel=in_channels*2, reduction=1)
	def forward(self, input):
		comp1=self.relu(self.map(self.batchnorm(self.conv3(input))))
		comp2=self.relu(self.avp(self.batchnorm(self.conv3(input))))
		comp3=self.relu(self.batchnorm(self.conv3s2(input)))
		comp4=self.relu(self.batchnorm(self.conv1s2(input)))
		input=torch.concatenate((comp1, comp2, comp3, comp4), dim=1)
		input=self.sca3d(input)
		return input

class Voxception(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv3=nn.Conv3d(in_channels, out_channels, 3, padding="same")
		self.conv1=nn.Conv3d(in_channels, out_channels, 1, padding="same")
		self.relu=nn.ELU()
	def forward(self, input):
		comp1=self.relu(self.conv3(input))
		comp2=self.relu(self.conv1(input))
		return torch.concatenate((comp1, comp2), dim=1)

class Voxception_ResNetLayer(nn.Module):
	def __init__(self, process_channels):
		super().__init__()
		self.path1=nn.Sequential(
			nn.Conv3d(process_channels, process_channels//2, 3, padding="same"),
			NormalizationLayer3d(process_channels//2),
			nn.ELU(),
			nn.Conv3d(process_channels//2, process_channels//2, 3, padding="same"),
			NormalizationLayer3d(process_channels//2),
			nn.ELU()
		)
		self.path2=nn.Sequential(
			nn.Conv3d(process_channels, process_channels//2, 1, padding="same"),
			NormalizationLayer3d(process_channels//2),
			nn.ELU(),
			nn.Conv3d(process_channels//2, process_channels//2, 3, padding="same"),
			NormalizationLayer3d(process_channels//2),
			nn.ELU(),
			nn.Conv3d(process_channels//2, process_channels//2, 1, padding="same"),
			NormalizationLayer3d(process_channels//2),
			nn.ELU()
		)
		self.sca3d=models.sca_3d.SCA3D(channel=process_channels, reduction=1)
		self.process_channels=process_channels
	def forward(self, input):
		input=input+torch.concatenate((self.path1(input), self.path2(input)), dim=1)
		input=self.sca3d(input)
		return input

class Voxception_Upsample(nn.Module): #Classic "Deconvolution" is not just unconvenient in terms of parameters and input size, but also has its issues https://distill.pub/2016/deconv-checkerboard/
	def __init__(self, in_channels, scale_factor):
		super().__init__()
		self.batchnorm=NormalizationLayer3d(in_channels//4)
		#self.conv3=nn.Conv3d(in_channels, in_channels//2, 3, padding="same") #TODO: Figure out corresponding Transpose3D
		#self.conv3s2=nn.Conv3d(in_channels, in_channels//2, 3, stride=2, padding=1)
		#self.conv1s2=nn.Conv3d(in_channels, in_channels//2, 1, stride=2)
		
		self.conv3=nn.Conv3d(in_channels, in_channels//4, 3, padding="same")
		self.conv3t1=nn.ConvTranspose3d(in_channels, in_channels//4, 3, stride=2, padding=1, output_padding=1)
		self.upsample=nn.Upsample(scale_factor=scale_factor)
		self.relu=nn.ELU()
		self.sca3d=models.sca_3d.SCA3D(channel=in_channels//2, reduction=1)
	def forward(self, input):
		comp1=self.relu(self.upsample(self.batchnorm(self.conv3(input))))
		comp2=self.relu(self.batchnorm(self.conv3t1(input)))
		input=torch.concatenate((comp1, comp2), dim=1)
		input=self.sca3d(input)
		#print("Downsample", input.shape, comp1.shape, comp2.shape, comp3.shape, comp4.shape, "->", ret.shape)
		return input

def Voxception_Block(N):
	return nn.Sequential(Voxception_ResNetLayer(N), Voxception_Downsample(N))

def Upsample_Block(N): #checkerboard artifacts (https://distill.pub/2016/deconv-checkerboard/) don't always occur but have been observed with this
	return nn.Sequential(Voxception_ResNetLayer(N), Voxception_Upsample(N, scale_factor=2))

class AttentionBlock(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.attn=nn.MultiheadAttention(dim, 4)
		self.bn=NormalizationLayer1d(dim) #Batch normalization is crucial here
	def forward(self, input):
		output, weights=self.attn(input, input, input)
		output+=input
		return output

class SamplingLayer(nn.Module):
	def __init__(self, latent_size, latent_depth, flatten, spacesize=1):
		super().__init__()
		self.N = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda")), torch.tensor(1.0).to(device=torch.device("cuda")))
		self.kl=torch.tensor(0.0)
		if flatten:
			dim=latent_size*latent_depth*latent_depth*latent_depth
			self.block_mean=nn.Sequential(nn.Linear(dim, dim), nn.ELU(), AttentionBlock(dim), nn.Linear(dim, dim), nn.ELU())
			self.block_var=nn.Sequential(nn.Linear(dim, dim), nn.ELU(), AttentionBlock(dim), nn.Linear(dim, dim), nn.ELU())
			self.flatten=nn.Flatten()
			self.unflatten=nn.Unflatten(1, (dim, 1, 1, 1))
		else:
			dim=latent_size
			self.block_mean=nn.Sequential(nn.Conv3d(dim, dim, 1), nn.ELU(), models.sca_3d.SCA3D(channel=dim, reduction=1), nn.Conv3d(dim, dim, 1), nn.ELU())
			self.block_var=nn.Sequential(nn.Conv3d(dim, dim, 1), nn.ELU(), models.sca_3d.SCA3D(channel=dim, reduction=1), nn.Conv3d(dim, dim, 1), nn.ELU())
			self.flatten=None
			self.unflatten=None
		self.perplexity=None
	def forward(self, input): #reference for the VAE part of the autoencoder: https://avandekleut.github.io/vae/
		input_shape=input.shape 
		if self.flatten is not None:
			input=self.flatten(input)
		mean=self.block_mean(input)
		var=torch.exp(self.block_var(input))
		self.kl = torch.sum(mean**2 + var**2 - torch.log(var) - 1/2)
		z=mean+var*self.N.sample(mean.shape)
		if self.flatten is not None:
			z=nn.Unflatten(1, input_shape[1:])(z)
		return z
	def to(self, *args, **kwargs):
		ret=super().to(*args, **kwargs)
		self.N.loc=self.N.loc.to(*args, **kwargs)
		self.N.scale=self.N.scale.to(*args, **kwargs)
		return ret

class SamplerVQ(nn.Module):
	def __init__(self, dim, vq_params):
		super().__init__()
		self.vq=models.vq_vae.VectorQuantizerEMA(vq_params["num_embeddings"], dim, vq_params["commitment_cost"], vq_params["decay"])
		self.vq_loss=0.0
		self.flatten=None
		self.unflatten=None
		self.encodings=None
	def forward(self, input):
		if self.flatten is not None:
			input=self.flatten(input)
		loss, ret, perplexity, encodings=self.vq(input)
		self.vq_loss=loss
		self.kl=self.vq_loss
		self.perplexity=perplexity.item()
		self.encodings=encodings.detach()
		if self.unflatten is not None:
			ret=self.unflatten(ret)
		return ret

def Bottleneck_Attn(bottleneck_channels, latent_dims, globalpool, is_3d, section_input): #global pool with 3D latent space should always be 1
	def _Block():
		if is_3d:
			return [models.sca_3d.SCA3D(channel=latent_dims, reduction=1),
			nn.Conv3d(latent_dims, latent_dims, 1, padding="same"),
			NormalizationLayer3d(latent_dims),
			nn.ELU()]
		else:
			return [AttentionBlock(latent_dims), nn.Linear(latent_dims, latent_dims), NormalizationLayer1d(latent_dims), nn.ELU()]
	def _Reshape_Block():
		if is_3d:
			if bottleneck_channels==latent_dims:
				return []
			return[
				nn.Conv3d(latent_dims, bottleneck_channels, 1, padding="same"),
				NormalizationLayer3d(bottleneck_channels),
				nn.ELU(),
				]
		if bottleneck_channels==latent_dims:
			return [nn.Unflatten(1, (bottleneck_channels, 1, 1, 1))]
		else:
			return [nn.Linear(latent_dims, bottleneck_channels), nn.ELU(), nn.Unflatten(1, (bottleneck_channels, 1, 1, 1))]
	if section_input:
		return nn.Sequential(
				DebugLayer("INTO BOTTLENECK"+str(globalpool)),
				nn.Conv3d(bottleneck_channels, latent_dims, 1, padding="same"),
				*([nn.AdaptiveMaxPool3d(globalpool if globalpool else 1)] if globalpool else []),
				DebugLayer("TO SOFTMAX"),
				nn.Softmax(dim=1),
				DebugLayer("TO SAMPLE"),
			)
	else:
		return nn.Sequential(
				DebugLayer("FROM SAMPLE"),
				DebugLayer("FROM BOTTLENECK LAST LAYER"),
				*_Reshape_Block(),
				DebugLayer("FROM BOTTLENECK"),
		)

def Bottleneck_VQVAE(bottleneck_channels, latent_dims, section_input):
	if section_input:
		return nn.Sequential(
				nn.Conv3d(bottleneck_channels, latent_dims, 1, padding="same"),
				models.sca_3d.SCA3D(channel=latent_dims, reduction=1),
				nn.Conv3d(latent_dims, latent_dims, 1, padding="same"),
				NormalizationLayer3d(latent_dims),
				nn.ELU(),
				models.sca_3d.SCA3D(channel=latent_dims, reduction=1),
				nn.Conv3d(latent_dims, latent_dims, 1, padding="same"),
				NormalizationLayer3d(latent_dims),
				nn.Softmax(dim=1)
		)
	else:
		return nn.Sequential(
				models.sca_3d.SCA3D(channel=latent_dims, reduction=1),
				nn.Conv3d(latent_dims, latent_dims, 1, padding="same"),
				NormalizationLayer3d(latent_dims),
				nn.ELU(),
				*([] if bottleneck_channels==latent_dims else [
					nn.Conv3d(latent_dims, bottleneck_channels, 1, padding="same"),
					NormalizationLayer3d(bottleneck_channels),
					nn.ELU(),
				]),
		)

class DebugLayer(nn.Module):
	def __init__(self, name, output_content=False):
		super().__init__()
		self.name=name
		self.output_content=output_content
	def forward(self, input):
		if not (torch.any(torch.isnan(input)) or torch.any(torch.isinf(input))):
			return input
		else:
			print("BAD VALUES:", input)
		print(self.name+":"+str(input.shape))
		if self.output_content:
			print(input)
		return input

def log2(x):
	return (torch.log(torch.tensor(x))/torch.log(torch.tensor(2))).item()

class PermuteLayer(nn.Module):
	def __init__(self, axes):
		super().__init__()
		self.axes=axes
	def forward(self, input):
		return input.permute(self.axes)

#latent dims: latent "dimensions" but in reality, the "depth" of the latent tensor (size of vector if 1d latent, otherwise "depth" of 4d tensor)

class Voxception_ResNet(nn.Module):
	def __init__(self, in_channels, latent_dims, base_channels=32, output_dim=32, encoder_depth=5, bottleneck_type=None, vq_param=None, attention=True, tensor_format="onehot"):
		super().__init__()
		bottleneck_channels=int(base_channels*(2**(encoder_depth)))
		latent_spacesize=4
		if bottleneck_type=="VAE": #VAE bottleneck
			self.Sampling=SamplingLayer(latent_dims, latent_spacesize, False)
			bottleneck1=Bottleneck_Attn(bottleneck_channels, latent_dims, latent_spacesize if attention!="3d" else 0, attention=="3d", True)
			bottleneck2=Bottleneck_Attn(bottleneck_channels, latent_dims, latent_spacesize if attention!="3d" else 0, attention=="3d", False)
			self.bottleneck_type="VAE"
		elif bottleneck_type=="VQ-VAE": #VQ-VAE bottleneck
			self.Sampling=SamplerVQ(latent_dims, vq_param)
			bottleneck1=Bottleneck_VQVAE(bottleneck_channels, latent_dims, True)
			bottleneck2=Bottleneck_VQVAE(bottleneck_channels, latent_dims, False)
			self.bottleneck_type="VQ-VAE"
		else: #Standard autoencoder bottleneck
			self.Sampling=nn.Identity()
			self.Sampling.perplexity=0.0
			bottleneck1=Bottleneck_Attn(bottleneck_channels, latent_dims, latent_spacesize if attention!="3d" else 0, attention=="3d", True)
			bottleneck2=Bottleneck_Attn(bottleneck_channels, latent_dims, latent_spacesize if attention!="3d" else 0, attention=="3d", False)
			self.bottleneck_type="AE"
		if tensor_format=="onehot":
			self.Encoder=nn.Sequential(
				DebugLayer("INTO ENCODER"),
				nn.Conv3d(in_channels, base_channels, 3, padding="same"),
				*[Voxception_Block(int(base_channels*(2**i))) for i in range(encoder_depth)],
				DebugLayer("FROM DOWNSAMPLE"),
				#nn.Softmax(dim=1),
				bottleneck1
			)
		else:
			embed_dim=4
			embed_vocab_size=255
			self.Encoder=nn.Sequential(
				DebugLayer("INTO ENCODER"),
				nn.Embedding(embed_vocab_size, embed_dim),
				PermuteLayer([0, 4, 1, 2, 3]),
				nn.Conv3d(embed_dim, base_channels, 3, padding="same"),
				*[Voxception_Block(int(base_channels*(2**i))) for i in range(encoder_depth)],
				DebugLayer("FROM DOWNSAMPLE"),
				#nn.Softmax(dim=1)
				bottleneck1
			)
		upsample_layers=encoder_depth
		us_parts=[bottleneck2]
		us_channels=bottleneck_channels
		for i in range(upsample_layers):
			us_parts.append(DebugLayer("Upsample["+str(i)+","+str(us_channels)+"]"))
			us_parts.append(Voxception_ResNetLayer(us_channels))
			us_parts.append(Upsample_Block(us_channels))
			us_channels//=2
		us_parts.append(DebugLayer("OUTPUT PREPROCESS"))
		us_parts.append(Voxception_ResNetLayer(us_channels))
		us_parts.append(nn.Conv3d(us_channels, in_channels, 1, stride=1, padding="same"))
		us_parts.append(nn.Softmax(dim=1))
		us_parts.append(DebugLayer("OUTPUT FINAL"))
		self.Decoder=nn.Sequential(*us_parts)
		del us_parts
		self.latent_dims=latent_dims
		def init_weights(m):
			if type(m) == nn.Linear:
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
		self.apply(init_weights)
		
	def forward(self, input):
		return self.Decoder(self.Sampling(self.Encoder(input)))
