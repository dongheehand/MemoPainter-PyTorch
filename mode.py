import os
import torch
import torch.nn as nn
from dataset import mydata
from torch.utils.data import DataLoader
import torch.optim as optim
from memory_network import Memory_Network
from generator import unet_generator
from discriminator import Discriminator
from util import zero_grad
from skimage.color import lab2rgb
import numpy as np
from PIL import Image


def train(args):
    
    model_path = os.path.join(args.model_path, args.data_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    train_log_path = '%s_train_log.txt'%(args.data_name)
    f = open(train_log_path, 'w')
    ### Logging configuration
    f.write('Data_name : %s \n'%(args.data_name))
    f.close()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Dataset and Dataloader for training
    tr_dataset = mydata(img_path = args.train_data_path, img_size = args.img_size, km_file_path = args.km_file_path, color_info = args.color_info)
    tr_dataloader = DataLoader(tr_dataset, batch_size = args.batch_size, shuffle=True, drop_last = True)
    
    if args.test_with_train:
        te_dataset = mydata(img_path = args.test_data_path, img_size = args.img_size,km_file_path = args.km_file_path, color_info = args.color_info)
        te_dataloader = DataLoader(te_dataset, batch_size = args.batch_size, shuffle=False, drop_last = False)
    
    ### Networks for coloring
    mem = Memory_Network(mem_size = args.mem_size, color_info = args.color_info, color_feat_dim = args.color_feat_dim, spatial_feat_dim = args.spatial_feat_dim, top_k = args.top_k, alpha = args.alpha).to(device)
    generator = unet_generator(args.input_channel, args.output_channel, args.n_feats, args.color_feat_dim).to(device)
    discriminator = Discriminator(args.input_channel + args.output_channel, args.color_feat_dim, args.img_size).to(device)
    
    generator = generator.train()
    discriminator = discriminator.train()
    
    ### Loss for coloring
    criterion_GAN = nn.BCELoss()
    criterion_sL1 = nn.SmoothL1Loss()
    
    ### Labels for adversarial training
    real_labels = torch.ones((args.batch_size, 1)).to(device)
    fake_labels = torch.zeros((args.batch_size, 1)).to(device)

    ### For Optimization
    g_opt = optim.Adam(generator.parameters(), lr = args.lr)
    d_opt = optim.Adam(discriminator.parameters(), lr = args.lr)
    m_opt = optim.Adam(mem.parameters(), lr = args.lr)
    opts = [g_opt, d_opt, m_opt]
    
    ### Training prcoess
    for e in range(args.epoch):
        for i, batch in enumerate(tr_dataloader):
            res_input = batch['res_input'].to(device)
            color_feat = batch['color_feat'].to(device)
            l_channel = (batch['l_channel'] / 100.0).to(device)
            ab_channel = (batch['ab_channel'] / 110.0).to(device)
            idx = batch['index'].to(device)

            ### 1) Train spatial feature extractor
            res_feature = mem(res_input)
            loss = mem.unsupervised_loss(res_feature, color_feat, args.color_thres)
            zero_grad(opts)
            loss.backward()
            m_opt.step()
            
            ### 2) Update Memory module
            with torch.no_grad():
                res_feature = mem(res_input)
                mem.memory_update(res_feature, color_feat, args.color_thres, idx)

            ### 3) Train Discriminator    
            dis_color_feat = torch.cat([torch.unsqueeze(color_feat, 2) for _ in range(args.img_size)], dim = 2)
            dis_color_feat = torch.cat([torch.unsqueeze(dis_color_feat, 3) for _ in range(args.img_size)], dim = 3)
            fake_ab_channel = generator(l_channel, color_feat)
            real = discriminator(ab_channel, l_channel, dis_color_feat)
            d_loss_real = criterion_GAN(real, real_labels)

            fake = discriminator(fake_ab_channel, l_channel, dis_color_feat)
            d_loss_fake = criterion_GAN(fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            
            zero_grad(opts)
            d_loss.backward()
            d_opt.step()
            
            ### 4) Train Generator
            fake_ab_channel = generator(l_channel, color_feat)
            fake = discriminator(fake_ab_channel, l_channel, dis_color_feat)
            g_loss_GAN = criterion_GAN(fake, real_labels)

            g_loss_smoothL1 = criterion_sL1(fake_ab_channel, ab_channel)
            g_loss = g_loss_GAN + g_loss_smoothL1

            zero_grad(opts)
            g_loss.backward()
            g_opt.step()
            
        f = open(train_log_path, 'a')
        f.write('%04d-epoch train loss'%(e))
        f.write('g_loss : %04f \t d_loss : %04f \n'%(g_loss.item(), d_loss.item()))
        f.close()
        
        if args.test_with_train and (e + 1) % args.test_freq  == 0:
            generator.eval()
            test_operation(args, generator, mem, te_dataloader, device, e)
            generator.train()
        
        if (e + 1) % args.model_save_freq == 0:
            torch.save(generator.state_dict(), os.path.join(model_path ,'generator_%03d.pt'%e))
            torch.save({'mem_model' : mem.state_dict(),
                       'mem_key' : mem.spatial_key.cpu(),
                       'mem_value' : mem.color_value.cpu(),
                       'mem_age' : mem.age.cpu(),
                       'mem_index' : mem.top_index.cpu()}, os.path.join(model_path, 'memory_%03d.pt'%e))


def test(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Dataset and Dataloader for test
    test_dataset = mydata(img_path = args.test_data_path, img_size = args.img_size, km_file_path = args.km_file_path, color_info = args.color_info)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)
    
    ### Networks for coloring
    mem = Memory_Network(mem_size = args.mem_size, color_info = args.color_info, color_feat_dim = args.color_feat_dim, spatial_feat_dim = 512, alpha = args.alpha)
    generator = unet_generator(args.input_channel, args.output_channel, args.n_feats, args.color_feat_dim)
    
    ### Load the pre-trained model
    mem_checkpoint = torch.load(args.mem_model)
    mem.load_state_dict(mem_checkpoint['mem_model'])
    mem.sptial_key = mem_checkpoint['mem_key']
    mem.color_value = mem_checkpoint['mem_value']
    mem.age = mem_checkpoint['mem_age']
    mem.top_index = mem_checkpoint['mem_index']
    
    generator.load_state_dict(torch.load(args.generator_model))

    mem.to(device)
    mem.spatial_key = mem.sptial_key.to(device)
    mem.color_value = mem.color_value.to(device)
    mem.age = mem.age.to(device)
    generator.to(device)
    
    generator = generator.eval()
    test_operation(args, generator, mem, test_dataloader, device)



def test_operation(args, generator, mem, te_dataloader, device, e = -1):
    
    count = 0
    result_path = os.path.join(args.result_path, args.data_name)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    
    with torch.no_grad():
        for i, batch in enumerate(te_dataloader):
            res_input = batch['res_input'].to(device)
            color_feat = batch['color_feat'].to(device)
            l_channel = (batch['l_channel'] / 100.0).to(device)
            ab_channel = (batch['ab_channel'] / 110.0).to(device)
            
            bs = res_input.size()[0]

            query = mem(res_input)
            top1_feature, _ = mem.topk_feature(query, 1)
            top1_feature = top1_feature[:, 0, :]
            result_ab_channel = generator(l_channel, top1_feature)
            
            real_image = torch.cat([l_channel * 100, ab_channel * 110], dim = 1).cpu().numpy()
            fake_image = torch.cat([l_channel * 100, result_ab_channel * 110], dim = 1).cpu().numpy()
            gray_image = torch.cat([l_channel * 100, torch.zeros((bs, 2, args.img_size, args.img_size)).to(device)], dim = 1).cpu().numpy()
            
            all_img = np.concatenate([real_image, fake_image, gray_image], axis = 2)
            all_img = np.transpose(all_img, (0, 2, 3, 1))
            rgb_imgs = [lab2rgb(ele) for ele in all_img]
            rgb_imgs = np.array((rgb_imgs))
            rgb_imgs = (rgb_imgs * 255.0).astype(np.uint8)
            
            for t in range(len(rgb_imgs)):
                
                if e > -1 :
                    img = Image.fromarray(rgb_imgs[t])
                    name = '%03d_%04d_result.png'%(e, count)
                    img.save(os.path.join(result_path, name))
                    
                else:
                    name = '%04d_%s.png'
                    img = rgb_imgs[t]
                    h, w, c = img.shape
                    stride = h // 3
                    original = img[:stride, :, :]
                    original = Image.fromarray(original)
                    original.save(os.path.join(result_path, name%(count, 'GT')))
                    
                    result = img[stride : 2*stride, :, :]
                    result = Image.fromarray(result)
                    result.save(os.path.join(result_path, name%(count, 'result')))
                    
                    if not args.test_only:
                        gray_img = img[2*stride :, :, :]
                        gray_img = Image.fromarray(gray_img)
                        gray_img.save(os.path.join(result_path, name%(count, 'gray')))

                count = count + 1

