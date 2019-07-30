import torch
import torch.nn as nn
from torch.nn import functional as F
from ResNet import ResNet18
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


class Memory_Network(nn.Module):
    
    def __init__(self, mem_size, color_info, color_feat_dim = 313, spatial_feat_dim = 512, top_k = 256, alpha = 0.1, age_noise = 4.0):
        
        super(Memory_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ResNet18 = ResNet18().to(self.device)
        self.ResNet18 = self.ResNet18.eval()
        self.mem_size = mem_size
        self.color_feat_dim = color_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        self.alpha = alpha
        self.age_noise = age_noise
        self.top_k = top_k
        self.color_info = color_info
        
        if self.color_info == 'dist':
            ## Each color_value is probability distribution
            self.color_value = F.normalize(random_uniform((self.mem_size, self.color_feat_dim), 0, 0.01), p = 1, dim=1).to(self.device)
        
        elif self.color_info == 'RGB':
            ## Each color_value is normalized RGB value [0, 1]
            self.color_value = random_uniform((self.mem_size, self.color_feat_dim), 0, 1).to(self.device)
        
        self.spatial_key = F.normalize(random_uniform((self.mem_size, self.spatial_feat_dim), -0.01, 0.01), dim=1).to(self.device)
        self.age = torch.zeros(self.mem_size).to(self.device)
        
        self.top_index = torch.zeros(self.mem_size).to(self.device)
        self.top_index = self.top_index - 1.0
        
        self.color_value.requires_grad = False
        self.spatial_key.requires_grad = False
        
        self.Linear = nn.Linear(512, spatial_feat_dim)
        self.body = [self.ResNet18, self.Linear]
        self.body = nn.Sequential(*self.body)
        
    def forward(self, x):
        q = self.body(x)
        q = F.normalize(q, dim = 1)
        return q
    
    def unsupervised_loss(self, query, color_feat, color_thres):
        
        bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        
        top_k_score, top_k_index = torch.topk(cosine_score, self.top_k, 1)
        
        ### For unsupervised training
        color_value_expand = torch.unsqueeze(torch.t(self.color_value), 0)
        color_value_expand = torch.cat([color_value_expand[:,:,idx] for idx in top_k_index], dim = 0)
        
        color_feat_expand = torch.unsqueeze(color_feat, 2)
        color_feat_expand = torch.cat([color_feat_expand for _ in range(self.top_k)], dim = 2)
        
        if self.color_info == 'dist':
            color_similarity = self.KL_divergence(color_value_expand, color_feat_expand, 1)
        
        elif self.color_info == 'RGB':
            color_similarity = self.CIEDE2000(color_value_expand.cpu().numpy(), color_feat_expand.cpu().numpy())
        
        loss_mask = color_similarity < color_thres
        loss_mask = loss_mask.float()
        
        pos_score, pos_index = torch.topk(torch.mul(top_k_score, loss_mask), 1, dim = 1)
        neg_score, neg_index = torch.topk(torch.mul(top_k_score, 1 - loss_mask), 1, dim = 1)
        
        loss = self._unsupervised_loss(pos_score, neg_score)
        
        return loss
    
    
    def memory_update(self, query, color_feat, color_thres, top_index):
        
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        top1_score, top1_index = torch.topk(cosine_score, 1, dim = 1)
        top1_index = top1_index[:, 0]
        top1_feature = self.spatial_key[top1_index]
        top1_color_value = self.color_value[top1_index]
        
        if self.color_info == 'dist':
            color_similarity = self.KL_divergence(top1_color_value, color_feat, 1)
            
        elif self.color_info == 'RGB':
            
            top1_color_value = top1_color_value.cpu().numpy()
            top1_color_value = np.expand_dims(top1_color_value, 2)
            color_feat_cpu = color_feat.cpu().numpy()
            color_feat_cpu = np.expand_dims(color_feat_cpu, 2)
            
            color_similarity = torch.squeeze(self.CIEDE2000(top1_color_value, color_feat_cpu))
            
        memory_mask = color_similarity < color_thres
        self.age = self.age + 1.0
        
        ## Case 1
        case_index = top1_index[memory_mask]
        self.spatial_key[case_index] = F.normalize(self.spatial_key[case_index] + query[memory_mask], dim = 1)
        self.age[case_index] = 0.0
        
        ## Case 2
        memory_mask = 1.0 - memory_mask
        case_index = top1_index[memory_mask]
        
        random_noise = random_uniform((self.mem_size, 1), -self.age_noise, self.age_noise)[:, 0]
        random_noise = random_noise.to(self.device)
        age_with_noise = self.age + random_noise
        old_values, old_index = torch.topk(age_with_noise, len(case_index), dim=0)
        
        self.spatial_key[old_index] = query[memory_mask]
        self.color_value[old_index] = color_feat[memory_mask]
        self.top_index[old_index] = top_index[memory_mask]
        self.age[old_index] = 0.0
        
    
    def topk_feature(self, query, top_k = 1):
        _bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        topk_score, topk_index = torch.topk(cosine_score, top_k, dim = 1)
        
        topk_feat = torch.cat([torch.unsqueeze(self.color_value[topk_index[i], :], dim = 0) for i in range(_bs)], dim = 0)
        topk_idx = torch.cat([torch.unsqueeze(self.top_index[topk_index[i]], dim = 0) for i in range(_bs)], dim = 0)
        
        return topk_feat, topk_idx

    
    def KL_divergence(self, a, b, dim, eps = 1e-8):
        
        b = b + eps
        log_val = torch.log10(torch.div(a, b))
        kl_div = torch.mul(a, log_val)
        kl_div = torch.sum(kl_div, dim = dim)
        
        return kl_div
    
    def CIEDE2000(self, color_value_expand, color_feat_expand):
        
        bs, color_dim, num_top_k = color_value_expand.shape
        
        color_value_expand = np.transpose(color_value_expand, (0, 2, 1))
        color_feat_expand = np.transpose(color_feat_expand, (0, 2, 1))
        
        color_value_expand = np.reshape(color_value_expand, (bs, num_top_k, 3, 10))
        color_feat_expand = np.reshape(color_feat_expand, (bs, num_top_k, 3, 10))
        
        color_value_expand = np.transpose(color_value_expand, (0, 1, 3, 2))
        color_feat_expand = np.transpose(color_feat_expand, (0, 1, 3, 2))
        
        color_sim = [deltaE_ciede2000(rgb2lab(color_value_expand[i]), rgb2lab(color_feat_expand[i])) for i in range(bs)]
        color_sim = np.mean(np.array(color_sim), axis = 2)
        
        color_sim = torch.tensor(color_sim).to(self.device)
        color_sim.requires_grad = False
        
        return color_sim
        
        
    def _unsupervised_loss(self, pos_score, neg_score):
        
        hinge = torch.clamp(neg_score - pos_score + self.alpha, min = 0.0)
        loss = torch.mean(hinge)
        
        return loss
        

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    result = (high - low) * x + low
    
    return result
    

