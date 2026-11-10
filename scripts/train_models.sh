# The following commands initate model training as it was conducted during research
python -u run_model_training.py -pt ground_truth -bs 32 -ne 200 > temp_experiment_ground_truth.log &\
python -u run_model_training.py -pt pinhole -bs 32 -ne 200 > temp_experiment_pinhole.log &\
python -u run_model_training.py -pt cylindrical -bs 32 -ne 200 > temp_experiment_cylindrical.log