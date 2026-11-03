# The following commands initate model training as it was conducted during research
python -u run_model_training.py --pt ground_truth --batch_size 32 --ne 50 &\
python -u run_model_training.py --pt pinhole --batch_size 32 --ne 50 &\
python -u run_model_training.py --pt cylindrical --batch_size 32 --ne 50