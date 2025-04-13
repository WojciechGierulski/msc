import os
import patoolib
import shutil
import glob

if not os.path.exists("datasets"):
    os.mkdir("datasets")
os.system("curl -0 -o datasets/data.rar https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar")
os.system("curl -0 -o datasets/labels.rar https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar")
if not os.path.exists("datasets/hmdb"):
    os.mkdir("datasets/hmdb")
    os.mkdir("datasets/hmdb/test_train_splits")
patoolib.extract_archive("datasets/labels.rar", outdir="datasets/hmdb/test_train_splits")
os.remove("datasets/labels.rar")
patoolib.extract_archive("datasets/data.rar", outdir="datasets/hmdb/data")
os.remove("datasets/data.rar")

for files in os.listdir("datasets/hmdb/data"):
    try:
        foldername = files.split(".")[0]
        os.mkdir(f"datasets/hmdb/data/{foldername}")
        patoolib.extract_archive(f"datasets/hmdb/data/{files}", outdir=f"datasets/hmdb/data")
    except FileExistsError:
        print(f"skipping {foldername}")

for f in glob.glob("datasets/hmdb/data/*.rar"):
    os.remove(f)
