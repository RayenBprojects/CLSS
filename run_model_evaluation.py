import argparse

from pathlib import Path

import os

from core import evaluate_model, compare_models


def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("-gm", "--ground_truth_model", default= "ground_truth_model.pt2")
    parser.add_argument("-pm", "--pinhole_model", default= "pinhole_model.pt2")
    parser.add_argument("-cm", "--cylindrical_model", default= "cylindrical_model.pt2")
    parser.add_argument("-p", "--print_batch_data", action= 'store_true')
    args= parser.parse_args()

    models_directory= Path("./models/saved")
    saved_models= next(os.walk(models_directory))[2]
    models_status= {
        "ground_truth": None,
        "pinhole": None,
        "cylindrical": None
    }

    if args.ground_truth_model in saved_models:
        models_status["ground_truth"]= True
    else:
        print("WARNING: ground truth model not found. Some evaluations will be unavailable.")
    
    if args.pinhole_model in saved_models:
        models_status["pinhole"]= True
    else:
        print("WARNING: pinhole model not found. Some evaluations will be unavailable.")
    
    if args.cylindrical_model in saved_models:
        models_status["cylindrical"]= True
    else:
        print("WARNING: cylindrical model not found. Some evaluations will be unavailable.")

    if models_status["ground_truth"]:
        print("-- Evaluating Ground Truth Model --")
        evaluate_model(model_name= args.ground_truth_model, projection_type= "ground_truth", batch_size= 1, print_batch_data= args.print_batch_data)

    if models_status["pinhole"]:
        print("-- Evaluating Pinhole Model Against Own Labels --")
        evaluate_model(model_name= args.pinhole_model, projection_type= "pinhole", batch_size= 1, print_batch_data= args.print_batch_data)

        print("-- Evaluating Pinhole Model Against Ground Truth labels --")
        evaluate_model(model_name= args.pinhole_model, projection_type= "pinhole", own_data= False, batch_size= 1, print_batch_data= args.print_batch_data)

    if models_status["ground_truth"] and models_status["pinhole"]:
        print("-- Comparing Pinhole Model Against Ground Truth Model")
        compare_models(truth_model_name= args.ground_truth_model, model_name= args.pinhole_model, projection_type= "pinhole", batch_size= 1, print_batch_data= args.print_batch_data)

    if models_status["cylindrical"]:
        print("-- Evaluating Cylindrical Model Against own Labels--")
        evaluate_model(model_name= args.cylindrical_model, projection_type= "cylindrical", batch_size= 1, print_batch_data= args.print_batch_data)

        print("-- Evaluating Cylindrical Model Against Ground Truth labels --")
        evaluate_model(model_name= args.cylindrical_model, projection_type= "cylindrical", own_data= False, batch_size= 1, print_batch_data= args.print_batch_data)
    
    if models_status["ground_truth"] and models_status["cylindrical"]:
        print("-- Comparing Cylindrical Model Against Ground Truth Model")
        compare_models(truth_model_name= args.ground_truth_model, model_name= args.cylindrical_model, projection_type= "cylindrical", batch_size= 1, print_batch_data= args.print_batch_data)

    print("-- Done --")
    

if __name__ == "__main__":
    main()