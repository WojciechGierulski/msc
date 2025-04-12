import os
import patoolib
import gdown
from constants import UCF101_CLASSES


if not os.path.exists("datasets"):
    os.mkdir("datasets")

id = '1l7Cg41Ojus8t1U85wU9ws4T3PTNou-OQ'
output = 'datasets/data.rar'
gdown.download(id=id, output=output)
patoolib.extract_archive("datasets/data.rar", outdir="datasets/")


## add classes to test splits
# path = "datasets/ucf101/test_train_splits/ucfTrainTestlist/"
# for file in ["testlist01.txt", "testlist02.txt", "testlist03.txt"]:
#     filepath = f"{path}{file}"
#     with open(filepath, "r") as file:
#         lines = file.readlines()
#     new_lines = []
#     for line in lines:
#         line = line.strip()
#         class_name = line.split("/")[0]
#         class_idx = UCF101_CLASSES.index(class_name)
#         new_line = f"{line} {class_idx}\n"
#         new_lines.append(new_line)
#     with open(filepath, "w") as file:
#         file.writelines(new_lines)