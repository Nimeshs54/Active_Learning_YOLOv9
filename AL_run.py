# """
# ACTIVE LEARNING ALGORITHM
# Require:
#      - Set of labeled samples L
#      - Miscellaneous samples that have not been labeled U
#      - Model initializes f0
#      - Active learning metric v

# Algorithm:
# 1. Divide U into batches
# 2. f <- f0
# 3. If U is still empty or has not met the stop condition then:
#      - Calculate scores for all batches of U using f
#      - U_best <- Batches highest score in U according to v
#      - Y_best <- Label U_best (person)
#      - Train f using L and (U_best, Y_best)
#      - U = U- U_best
#      - L = L + (U_best, y_best)
# """

# import os
# import subprocess
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# from torch.cuda.memory import reset_accumulated_memory_stats
# from AL_yolov9 import Yolov9
# import AL_config as config
# import glob
# from shutil import copyfile, move
# import io
# import copy
# import random

# # try:
# #     import comet_ml  # must be imported before torch (if installed)
# # except ImportError:
# #     comet_ml = None


# def RandomSelect(num_select, result):
#     return random.sample(result.keys(), num_select)

# def UncertaintySamplingBinary(num_select, result, typ):
#     """
#     result = 
#         {"<link image>": 
#             [
#                 {"class": cls.item(), "box": [x,y,w,h], "conf": conf.item(),
#                 ...
#             ],
#         ...
#         }
#     """
#     probas = {}
#     if typ == 'sum':
#         for item, lst_dic in result.items():
#             conf = 0
#             for dic in lst_dic:
#                 conf += (1.0 - dic["conf"])
#             probas[item] = conf
#     elif typ == 'avg':
#         for item, lst_dic in result.items():
#             conf = 0
#             for dic in lst_dic:
#                 conf += (1.0 - dic["conf"])
#             probas[item] = conf/len(lst_dic)
#     elif typ == 'max':
#         for item, lst_dic in result.items():
#             conf = 0
#             for dic in lst_dic:
#                 conf = max(conf, 1.0 - dic["conf"])
#             probas[item] = conf
#     return sorted(probas, key=probas.get, reverse=True)[:num_select]


# class ActiveLearning(object):
#     def __init__(self, model):
#         self.model = model
#         self.num_select = config.num_select
#         self.type = 'sum' # 'avg' , 'max', 'sum'

#     def run(self):
#         # query number
#         queried = 0
#         ep = 1
#         # If there are not enough queries, continue to query
#         while queried < config.max_queried:
#             print("SECOND QUERY: ", queried)
#             # Predict images in the unlabeled set
#             result = self.model.detect()

#             # Select the k images with the highest score
#             # Use uncertainty sampling
#             if len(result) >= self.num_select:
#                 # U_best = RandomSelect(self.num_select, result)
#                 U_best = UncertaintySamplingBinary(self.num_select, result, 'sum')
#                 print("\n")
#                 print("The selected k images using Uncertainty Sampling: ", U_best)
                
#                 # Label files in samples (Interactive)

#                 # # Browse all selected files
#                 # for f in U_best:
#                 #     # Move the image file into the labeled folder
#                 #     move(f, f.replace("unlabeled", "labeled"))
#                 #     # Create a label file in the labeled folder
#                 #     type_file = f.split('.')[-1]
#                 #     copyfile(f.replace("unlabeled","truck_real_pool").replace(type_file, 'txt'), f.replace("unlabeled", "labeled").replace(type_file, 'txt'))

#                 # Browse all selected files
#                 for f in U_best:
#                     # Move the image file into the labeled folder
#                     move(f, f.replace("unlabeled", "labeled"))
                    
#                     # Extract the directory and filename
#                     directory, filename = os.path.split(f)
                    
#                     # Split the filename into base and extension
#                     file_base, file_extension = os.path.splitext(filename)
                    
#                     # Construct the new filename with .txt extension
#                     new_filename = file_base + '.txt'
                    
#                     # Construct new paths for the copy operation
#                     source_path = os.path.join(directory.replace("unlabeled", "voc_pool"), new_filename)
#                     destination_path = os.path.join(directory.replace("unlabeled", "labeled"), new_filename)
                    
#                     # Create a label file in the labeled folder
#                     copyfile(source_path, destination_path)


#                 # Add a list of labeled files to the training data
#                 with open('dataset/pascal_voc/train.txt', "a") as f:
#                     for file_name in U_best:
#                         f.write(file_name.replace("unlabeled","labeled") + '\n')

#                 # Train model
#                 self.model.train(ep)

#                 ####################### LOADING ########################
#                 # Delete the old weight file
#                 if os.path.exists(config.weight):
#                     os.remove(config.weight)

#                 # Update new weight
#                 copyfile(os.path.join(config.project_train, config.name, 'weights', 'best.pt'), config.weight)

#                 # Perform validation
#                 print("-------------------------------------------------Result on Test Set------------------------------------------")
#                 self.validate_model(queried)
#                 print("--------------------------------------------------Test Set End-----------------------------------------------")
                
#                 queried+=1
#                 ep += config.epochs
#             else:
#                 print("The number of unlabeled files are not enough {} files".format(self.num_select))
#                 break

#         # Save all test results to file after completing the queries
#         with open("test_results.txt", "w") as file:
#             for result in self.test_results:
#                 file.write(result + '\n')

#     def validate_model(self, queried):
#         # Construct the validation command
#         command = [
#             "python", "val_dual.py",
#             "--data", "dataset/pascal_voc/VOC2007.yaml",
#             "--weights", "runs/train/voc2007/weights/best.pt",
#             "--device", "0",
#             "--task", "test",
#             "--name", "voc2007_test"
#         ]

#         # Run the validation command
#         result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
#         # Print the output of the validation script
#         print(result.stdout)
#         print(result.stderr)

#         # Append the results to the list
#         self.test_results.append(f"Query {queried} Results:\n{result.stdout}")

# if __name__ == '__main__':
#     bot = ActiveLearning(model=Yolov9())
#     bot.run()




import os
import subprocess
import logging
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from torch.cuda.memory import reset_accumulated_memory_stats
from AL_yolov9 import Yolov9  # Assuming you have this module
import AL_config as config
import glob
from shutil import copyfile, move
import io
import copy
import random

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("all_logs.txt"), logging.StreamHandler()])

def RandomSelect(num_select, result):
    return random.sample(result.keys(), num_select)

def UncertaintySamplingBinary(num_select, result, typ):
    """
    result = 
        {"<link image>": 
            [
                {"class": cls.item(), "box": [x,y,w,h], "conf": conf.item(),
                ...
            ],
        ...
        }
    """
    probas = {}
    if typ == 'sum':
        for item, lst_dic in result.items():
            conf = 0
            for dic in lst_dic:
                conf += (1.0 - dic["conf"])
            probas[item] = conf
    elif typ == 'avg':
        for item, lst_dic in result.items():
            conf = 0
            if len(lst_dic) > 0:
                for dic in lst_dic:
                    conf += (1.0 - dic["conf"])
                probas[item] = conf / len(lst_dic)
            else:
                probas[item] = 0  # Or some other value to represent the uncertainty when there are no detections
    elif typ == 'max':
        for item, lst_dic in result.items():
            conf = 0
            for dic in lst_dic:
                conf = max(conf, 1.0 - dic["conf"])
            probas[item] = conf
    return sorted(probas, key=probas.get, reverse=True)[:num_select]
    # probas = {}
    # if typ == 'sum':
    #     for item, lst_dic in result.items():
    #         conf = 0
    #         for dic in lst_dic:
    #             conf += (1.0 - dic["conf"])
    #         probas[item] = conf
    # elif typ == 'avg':
    #     for item, lst_dic in result.items():
    #         conf = 0
    #         for dic in lst_dic:
    #             conf += (1.0 - dic["conf"])
    #         probas[item] = conf/len(lst_dic)
    # elif typ == 'max':
    #     for item, lst_dic in result.items():
    #         conf = 0
    #         for dic in lst_dic:
    #             conf = max(conf, 1.0 - dic["conf"])
    #         probas[item] = conf
    # return sorted(probas, key=probas.get, reverse=True)[:num_select]


class ActiveLearning(object):
    def __init__(self, model):
        self.model = model
        self.num_select = config.num_select
        self.type = 'avg' # 'avg' , 'max', 'sum'
        self.test_results = []

    def run(self):
        # query number
        queried = 0
        ep = 1
        # If there are not enough queries, continue to query
        while queried < config.max_queried:
            logging.info("SECOND QUERY: %d", queried)
            # Predict images in the unlabeled set
            result = self.model.detect()

            # Select the k images with the highest score
            # Use uncertainty sampling
            if len(result) >= self.num_select:
                # U_best = RandomSelect(self.num_select, result)
                U_best = UncertaintySamplingBinary(self.num_select, result, 'avg')
                logging.info("The selected k images using Uncertainty Sampling: %s", U_best)
                
                # Label files in samples (Interactive)

                # # Browse all selected files
                # for f in U_best:
                #     # Move the image file into the labeled folder
                #     move(f, f.replace("unlabeled", "labeled"))
                #     # Create a label file in the labeled folder
                #     type_file = f.split('.')[-1]
                #     copyfile(f.replace("unlabeled","truck_real_pool").replace(type_file, 'txt'), f.replace("unlabeled", "labeled").replace(type_file, 'txt'))

                # Browse all selected files
                for f in U_best:
                    # Move the image file into the labeled folder
                    move(f, f.replace("unlabeled", "labeled"))
                    
                    # Extract the directory and filename
                    directory, filename = os.path.split(f)
                    
                    # Split the filename into base and extension
                    file_base, file_extension = os.path.splitext(filename)
                    
                    # Construct the new filename with .txt extension
                    new_filename = file_base + '.txt'
                    
                    # Construct new paths for the copy operation
                    source_path = os.path.join(directory.replace("unlabeled", "truck_real_pool"), new_filename)
                    destination_path = os.path.join(directory.replace("unlabeled", "labeled"), new_filename)
                    
                    # Create a label file in the labeled folder
                    copyfile(source_path, destination_path)

                # Add a list of labeled files to the training data
                with open('dataset/TRUCK_Real/train.txt', "a") as f:
                    for file_name in U_best:
                        f.write(file_name.replace("unlabeled","labeled") + '\n')

                # Train model
                self.model.train(ep)

                ####################### LOADING ########################

                # Store the old weight file instead of deleting it
                # Check if 'best.pt' exists
                # Rename the old 'best.pt' file if it exists
                if os.path.exists(config.weight):
                    # Increment the names of existing files
                    counter = 1
                    while os.path.exists(f"{os.path.splitext(config.weight)[0]}{counter}.pt"):
                        counter += 1

                    # Rename the current 'best.pt' to 'bestN.pt'
                    os.rename(config.weight, f"{os.path.splitext(config.weight)[0]}{counter}.pt")

                # Save the new weight as 'best.pt'
                copyfile(os.path.join(config.project_train, config.name, 'weights', 'best.pt'), config.weight)

                # # Delete the old weight file
                # if os.path.exists(config.weight):
                #     os.remove(config.weight)

                # # Update new weight
                # copyfile(os.path.join(config.project_train, config.name, 'weights', 'best.pt'), config.weight)

                # Perform validation
                logging.info("-------------------------------------------------Result on Test Set------------------------------------------")
                self.validate_model(queried)
                logging.info("--------------------------------------------------Test Set End-----------------------------------------------")
                
                queried += 1
                ep += config.epochs
            else:
                logging.info("The number of unlabeled files are not enough %d files", self.num_select)
                break

        # Save all test results to file after completing the queries
        with open("test_results.txt", "w") as file:
            for result in self.test_results:
                file.write(result + '\n')

    def validate_model(self, queried):
        # Construct the validation command
        command = [
            "python", "val_dual.py",
            "--data", "dataset/TRUCK_Real/truck_data.yaml",
            "--weights", "runs/train/TRUCK_Real/weights/best.pt",
            "--device", "0",
            "--task", "test",
            "--name", "TRUCK_Real_test"
        ]

        # Run the validation command
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        # Print the output of the validation script
        logging.info(result.stdout)
        logging.error(result.stderr)

        # Append the results to the list
        self.test_results.append(f"Query {queried} Results:\n{result.stdout}")

if __name__ == '__main__':
    bot = ActiveLearning(model=Yolov9())
    bot.run()

