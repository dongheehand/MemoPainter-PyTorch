#### Training script

### Color feature : color distribution
# For pokemon
# python main.py --color_info dist --color_feat_dim 313 --mem_size 982 --top_k 256 --train_data_path ./pokemon --test_data_path ./test_data --alpha 0.3 --color_thres 0.5 --epoch 100

# For oxford102
# python main.py --color_info dist --color_feat_dim 313 --mem_size 7861 --train_data_path ./oxford102/train --test_data_path ./oxford102/temp_test --epoch 30 --model_save_freq 2 --color_thres 0.8 --data_name Flower

### Color feature : RGB values
# For pokemon
# python main.py --color_info RGB --color_feat_dim 30 --color_thres 8 --train_data_path ./pokemon/ --test_data_path ./test_data --epoch 100 --alpha 0.3 --top_k 32 --data_name pokemon

# For oxford102
# python main.py --color_info RGB --color_feat_dim 30 --color_thres 10 --train_data_path ./oxford102/train --test_data_path ./oxford102/temp_test --epoch 30 --alpha 0.1 --top_k 32 --mem_size 7861 --model_save_freq 2 --data_name Flower

### Test script

### Color feature : color distribution
# For pokemon
# python main.py --color_info dist --color_feat_dim 313 --mem_size 982 --test_data_path ./test_data/ --mem_model ./model/pokemon/memory_098.pt --generator_model ./model/pokemon/generator_098.pt --data_name poke_test --mode test

### Color featre : RGB values
# For pokemon
python main.py --color_feat_dim 30 --color_info RGB --test_data_path ./pokemon_test/ --data_name poke_test --mem_model ./model/pokemon/memory_099.pt --generator_model ./model/pokemon/generator_099.pt --mode test

# For oxford102
python main.py --color_feat_dim 30 --color_info RGB --test_data_path ./oxford102/test/ --data_name flower_test --mem_model ./model/Flower/memory_028.pt --genarator_model ./model/Flower/generator_028.pt --mode test

