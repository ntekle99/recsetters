TOKENIZERS_PARALLELISM=False python3 main.py --dataset=tracks --dataset_dir=/Users/bibi/Desktop/USC/AI:ML/final-project-recsetters/data_collection/ --device=cpu --batch_size=512 --print_freq=32 --lr=2e-4 --epochs=2 --margin=1 --num_negatives=20 --warm_threshold=0.2 --num_workers=8