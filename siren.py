import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()



        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        return torch.sin(freq * x + phase_shift)


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.is_siren=True

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor

class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.is_siren=True

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        
        return torch.cat([rbg, sigma], dim=-1)
    
    
    
class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()
    
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=True)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""
    
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.is_siren=True

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(32 + 3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        print(self.network)

        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)

    
    
class EmbeddingPiGAN256(EmbeddingPiGAN128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64)*0.1)



def gauss_act(x, sigma):
    # print("before (", sigma, "): ", x)
    x = torch.exp(-0.5*torch.square(x/sigma))
    # print("after (", sigma, "): ", x)
    return x

#From CodeNeRF
def PE(x, degree):
    y = torch.cat([2.**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)

class GaussLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma=1):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.sigma = sigma #nn.Parameter(torch.tensor(0.1), requires_grad=True)


    def forward(self, x):
        x = self.layer(x)
        return gauss_act(x, self.sigma)

class ReLULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer(x)
        return self.relu(x)

class FiLMLayer_Gauss(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma=1, z_dim=100):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.sigma = sigma
        self.cond_layer = nn.Linear(z_dim, 2)

    def forward(self, x, z):
        x = self.layer(x)
        y = self.cond_layer(z)
        a = y[:,0].unsqueeze(1).unsqueeze(1)
        b = y[:,1].unsqueeze(1).unsqueeze(1)

        # print("***************")
        # print(a.shape,b.shape,x.shape) #torch.Size([18]) torch.Size([18]) torch.Size([18, 12288, 256])


        return gauss_act(x, self.sigma)*a + b


class MLP_Gauss(nn.Module):

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()

        self.is_siren=False

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.network = nn.ModuleList([
            GaussLayer(3+z_dim, hidden_dim, sigma=0.15),
            GaussLayer(z_dim, hidden_dim, sigma=0.25),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15), 
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25), 
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = GaussLayer(hidden_dim + 3, hidden_dim, sigma=0.2)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        input = self.gridwarper(input)

        # print(input.shape, z.shape) #torch.Size([28, 12288, 3]) torch.Size([28, 12288, 3]) torch.Size([28, 256])

        z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        x = torch.cat([input,z_exp],dim=-1)


        # print("in :", x)
        for index, layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            x = layer(x)#, frequencies[..., start:end], phase_shifts[..., start:end])
            # print(index, " :", x)
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1))#, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        # print("x: ", x)
        # print("sig: ", sigma)
        # print("rgb: ", rbg)
        
        return torch.cat([rbg, sigma], dim=-1)

class MLP_Gauss_add(nn.Module):

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()

        self.is_siren=False

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.encoding_xyz = GaussLayer(3,z_dim, sigma=0.2)#0.12)
        self.encoding_dir = GaussLayer(3,hidden_dim, sigma=0.2)#0.12)

        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.network = nn.ModuleList([
            # GaussLayer(3+z_dim, hidden_dim),
            GaussLayer(z_dim, hidden_dim, sigma=0.25),#0.16), #this one was wrong! :(
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),#0.13),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),#0.11),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),#0.09),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),#0.06),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15), #
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25), #
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = GaussLayer(hidden_dim, hidden_dim, sigma=0.2)#0.15)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        input = self.gridwarper(input)

        # print(input.shape, z.shape) #torch.Size([28, 12288, 3]) torch.Size([28, 12288, 3]) torch.Size([28, 256])

        y = self.encoding_xyz(input)
        z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)


        x = y + z_exp

        # print("y: ", y)
        # print("z: ", z_exp)
        # print("x: ", x)

        viewdir = self.encoding_dir(ray_directions)


            
        for index, layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            x = layer(x)#, frequencies[..., start:end], phase_shifts[..., start:end])

        # print(x)
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(x+viewdir)#torch.cat([viewdir, x], dim=-1))#self.color_layer_sine(viewdir + x)#, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        # print("x: ", x)
        # print("sig: ", sigma)
        # print("rgb: ", rbg)
        
        return torch.cat([rbg, sigma], dim=-1)


class MLP_Gauss_FiLM(nn.Module):

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()

        self.is_siren=False

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        

        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.network = nn.ModuleList([
            FiLMLayer_Gauss(3, hidden_dim, sigma=0.2, z_dim=z_dim),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.16), 
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.13),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.11),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.09),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.06),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim), #
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim), #
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2, z_dim=z_dim)#0.15)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer_Gauss(hidden_dim+3, hidden_dim, sigma=0.2, z_dim=z_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        input = self.gridwarper(input)

        # print(input.shape, z.shape) #torch.Size([28, 12288, 3]) torch.Size([28, 12288, 3]) torch.Size([28, 256])

        x = input
            
        for index, layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            x = layer(x,z)#, frequencies[..., start:end], phase_shifts[..., start:end])

        # print(x)
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1),z)#self.color_layer_sine(viewdir + x)#, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        # print("x: ", x)
        # print("sig: ", sigma)
        # print("rgb: ", rbg)
        
        return torch.cat([rbg, sigma], dim=-1)

class MLP_Gauss_FiLM_PE(nn.Module):

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.num_xyz_freq = 10
        self.num_dir_freq = 4
        d_xyz, d_viewdir = 3 + 6 * self.num_xyz_freq, 3 + 6 * self.num_dir_freq

        self.is_siren=False

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoding_xyz = GaussLayer(d_xyz, hidden_dim, sigma=0.2) #nn.Sequential(nn.Linear(d_xyz, hidden_dim), nn.ReLU())

        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.network = nn.ModuleList([
            # FiLMLayer_Gauss(3, hidden_dim, sigma=0.2, z_dim=z_dim),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.16), #this one was wrong! :(
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.13),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.11),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.09),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim),#0.06),
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim), #
            FiLMLayer_Gauss(hidden_dim, hidden_dim, sigma=0.2,z_dim=z_dim), #
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer_Gauss(hidden_dim + d_viewdir, hidden_dim, sigma=0.2,z_dim=z_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        input = self.gridwarper(input)
        input = PE(input, self.num_xyz_freq)

        # print(input.shape, z.shape) #torch.Size([28, 12288, 3]) torch.Size([28, 12288, 3]) torch.Size([28, 256])

        # z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        # x = torch.cat([input,z_exp],dim=-1)

        x = self.encoding_xyz(input)
        # z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        # x = y + z_exp

        # print(input.shape)

        viewdir = PE(ray_directions, self.num_dir_freq)

            
        for index, layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            x = layer(x,z)#, frequencies[..., start:end], phase_shifts[..., start:end])

            # print(index, x)

        # print(x)

        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([viewdir, x], dim=-1),z)#, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        
        return torch.cat([rbg, sigma], dim=-1)


class MLP_Gauss_PE(nn.Module):

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.num_xyz_freq = 10
        self.num_dir_freq = 4
        d_xyz, d_viewdir = 3 + 6 * self.num_xyz_freq, 3 + 6 * self.num_dir_freq

        self.is_siren=False

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoding_xyz = GaussLayer(d_xyz, z_dim, sigma=0.2) #nn.Sequential(nn.Linear(d_xyz, hidden_dim), nn.ReLU())

        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.network = nn.ModuleList([
            # GaussLayer(3+z_dim, hidden_dim),
            GaussLayer(z_dim, hidden_dim, sigma=0.25),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.15),
            GaussLayer(hidden_dim, hidden_dim, sigma=0.25),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = GaussLayer(hidden_dim + d_viewdir, hidden_dim, sigma=0.2)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        input = self.gridwarper(input)
        input = PE(input, self.num_xyz_freq)

        # print(input.shape, z.shape) #torch.Size([28, 12288, 3]) torch.Size([28, 12288, 3]) torch.Size([28, 256])

        # z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        # x = torch.cat([input,z_exp],dim=-1)

        y = self.encoding_xyz(input)
        z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        x = y + z_exp

        # print(input.shape)

        viewdir = PE(ray_directions, self.num_dir_freq)

            
        for index, layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            x = layer(x)#, frequencies[..., start:end], phase_shifts[..., start:end])

            # print(index, x)

        # print(x)

        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([viewdir, x], dim=-1))#, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        
        return torch.cat([rbg, sigma], dim=-1)



class MLP_ReLU(nn.Module):

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.num_xyz_freq = 10
        self.num_dir_freq = 4
        d_xyz, d_viewdir = 3 + 6 * self.num_xyz_freq, 3 + 6 * self.num_dir_freq

        self.is_siren=False

        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, hidden_dim), nn.ReLU())

        self.mapping_network = [] #CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) #hacky thing to let work with existing architecture... Not sure if the averaging in actual implementation is a problem we need to address?
        
        self.network = nn.ModuleList([
            # ReLULayer(3, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = ReLULayer(hidden_dim + d_viewdir, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        input = self.gridwarper(input)
        input = PE(input, self.num_xyz_freq)

        # print(input.shape, z.shape) #torch.Size([28, 12288, 3]) torch.Size([28, 12288, 3]) torch.Size([28, 256])

        # z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        # x = torch.cat([input,z_exp],dim=-1)

        y = self.encoding_xyz(input)
        z_exp = z.reshape(z.shape[0],1,z.shape[1]).repeat(1,input.shape[1],1)
        x = y + z_exp

        # print(input.shape)

        viewdir = PE(ray_directions, self.num_dir_freq)




            
        for index, layer in enumerate(self.network):
            # start = index * self.hidden_dim
            # end = (index+1) * self.hidden_dim
            x = layer(x)#, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([viewdir, x], dim=-1))#, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        # print("x: ", x)
        # print("sig: ", sigma)
        # print("rgb: ", rbg)
        
        return torch.cat([rbg, sigma], dim=-1)

