import os
import patoolib
import shutil
import glob

if not os.path.exists("datasets"):
    os.mkdir("datasets")
os.system("wget datasets/data.rar https://www.crcv.ucf.edu/data/UCF101/UCF101.rar")
os.system("curl -o datasets/UCF101TrainTestSplits-RecognitionTask.zip https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
if not os.path.exists("datasets/ucf101"):
    os.mkdir("datasets/ucf101")
if not os.path.exists("datasets/ucf101/test_train_splits"):
    os.mkdir("datasets/ucf101/test_train_splits")
patoolib.extract_archive("datasets/UCF101TrainTestSplits-RecognitionTask.zip", outdir="datasets/ucf101/test_train_splits")
os.remove("datasets/UCF101TrainTestSplits-RecognitionTask.zip")
patoolib.extract_archive("datasets/data.rar", outdir="datasets/ucf101/data")
os.remove("datasets/data.rar")