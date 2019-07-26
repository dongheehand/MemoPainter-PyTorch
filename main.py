from mode import *
import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

## Model specification
parser.add_argument("--input_channel", type = int, default = 1)
parser.add_argument("--output_channel", type = int, default = 2)
parser.add_argument("--n_feats", type = int, default = 64)
parser.add_argument("--color_feat_dim", type = int, default = 313)
parser.add_argument("--spatial_feat_dim", type = int, default = 512)
parser.add_argument("--mem_size", type = int, default = 982)
parser.add_argument("--alpha", type = float, default = 0.1)
parser.add_argument("--top_k", type = int, default = 256)
parser.add_argument("--color_info", type = str, default = 'dist', help = 'option should be dist or RGB')

## Data specification 
parser.add_argument("--train_data_path", type = str, default = './pokemon/')
parser.add_argument("--test_data_path", type = str, default = './pokemon/')
parser.add_argument("--data_name", type = str, default = 'pokemon')
parser.add_argument("--km_file_path", type = str, default = './pts_in_hull.npy')
parser.add_argument("--img_size", type = int, default = 256)
parser.add_argument("--model_path", type = str, default = './model')
parser.add_argument("--result_path", type = str, default = './result')
parser.add_argument("--mem_model", type = str)
parser.add_argument("--generator_model", type = str)

## Training or test specification
parser.add_argument("--batch_size", type = int, default = 4)
parser.add_argument("--mode", type = str, default = 'train')
parser.add_argument("--lr", type = float, default = 1e-4)
parser.add_argument("--epoch", type = int, default = 50)
parser.add_argument("--color_thres", type = float, default = 0.7)
parser.add_argument("--test_with_train", type = str2bool, default = True)
parser.add_argument("--test_freq", type = int, default = 2)
parser.add_argument("--model_save_freq", type = int, default = 1)
parser.add_argument("--test_only", type = str2bool, default = False)

args = parser.parse_args()

if args.mode == 'train':
    train(args)

elif args.mode == 'test':
    test(args)

