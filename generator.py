## Some parts related to AdaIN are borrowed from https://github.com/NVlabs/MUNIT/ ##

import torch
import torch.nn as nn
import torch.nn.functional as F


class unet_generator(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf, color_dim = 313):
        super(unet_generator, self).__init__()
        
        self.e1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.e2 = unet_encoder_block(ngf, ngf * 2)
        self.e3 = unet_encoder_block(ngf * 2, ngf * 4)
        self.e4 = unet_encoder_block(ngf * 4, ngf * 8)
        self.e5 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e6 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e7 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e8 = unet_encoder_block(ngf * 8, ngf * 8, norm = None)

        self.d1 = unet_decoder_block(ngf * 8, ngf * 8)
        self.d2 = unet_decoder_block(ngf * 8 * 2, ngf * 8)
        self.d3 = unet_decoder_block(ngf * 8 * 2, ngf * 8)
        self.d4 = unet_decoder_block(ngf * 8 * 2, ngf * 8, drop_out = None)
        self.d5 = unet_decoder_block(ngf * 8 * 2, ngf * 4, drop_out = None)
        self.d6 = unet_decoder_block(ngf * 4 * 2, ngf * 2, drop_out = None)
        self.d7 = unet_decoder_block(ngf * 2 * 2, ngf, drop_out = None)
        self.d8 = unet_decoder_block(ngf * 2, output_nc, norm = None, drop_out = None)
        self.tanh = nn.Tanh()
        
        self.layers = [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8,
                 self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7, self.d8]
        
        self.mlp = MLP(color_dim, self.get_num_adain_params(self.layers), self.get_num_adain_params(self.layers), 3)

    
    def forward(self, x, color_feat):
        
        ### AdaIn params
        adain_params = self.mlp(color_feat)
        self.assign_adain_params(adain_params, self.layers)
        
        ### Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        ### Decoder
        d1_ = self.d1(e8)
        d1 = torch.cat([d1_, e7], dim = 1)
        
        d2_ = self.d2(d1)
        d2 = torch.cat([d2_, e6], dim = 1)
        
        d3_ = self.d3(d2)
        d3 = torch.cat([d3_, e5], dim = 1)
        
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e4], dim = 1)
        
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e3], dim = 1)
        
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e2], dim = 1)
        
        d7_ = self.d7(d6)
        d7 = torch.cat([d7_, e1], dim = 1)
        
        d8 = self.d8(d7)
        
        output = self.tanh(d8)
        
        return output
    
    def get_num_adain_params(self, _module):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    num_adain_params += 2*m.num_features
        return num_adain_params
    
    def assign_adain_params(self, adain_params, _module):
        # assign the adain_params to the AdaIN layers in model
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2*m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    if adain_params.size(1) > 2*m.num_features:
                        adain_params = adain_params[:, 2*m.num_features:]
        
        
class unet_encoder_block(nn.Module):
    
    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.LeakyReLU(inplace = True, negative_slope = 0.2)):
        super(unet_encoder_block, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, ks, stride, padding)
        m = [act, self.conv]
        
        if norm == 'adain':
            m.append(AdaptiveInstanceNorm2d(output_nc))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        return self.body(x)
    
class unet_decoder_block(nn.Module):
    
    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.ReLU(inplace = True), drop_out = 0.5):
        super(unet_decoder_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_nc, output_nc, ks, stride, padding)
        m = [act, self.deconv]
        
        if norm == 'adain':
            m.append(AdaptiveInstanceNorm2d(output_nc))
            
        if drop_out is not None:
            m.append(nn.Dropout(drop_out))
            
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        return self.body(x)
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, act = nn.ReLU(inplace = True)):

        super(MLP, self).__init__()
        self.model = []
        
        self.model.append(nn.Linear(input_dim, dim))
        self.model.append(act)
        
        for i in range(n_blk - 2):
            self.model.append(nn.Linear(dim, dim))
            self.model.append(act)
            
        self.model.append(nn.Linear(dim, output_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

