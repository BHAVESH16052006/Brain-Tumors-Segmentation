import os
import shutil
import random

SOURCE = r"C:\Users\chand\BrainTumorProject\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
DEST = r"C:\Users\chand\BrainTumorProject\dataset\brats2023"

TRAIN = 833
VAL = 209
TEST = 211

random.seed(42)

patients = [
    x for x in os.listdir(SOURCE)
    if os.path.isdir(os.path.join(SOURCE, x))
    and x.startswith("BraTS-GLI-")
]

print("Total patients found:", len(patients))

random.shuffle(patients)

train_patients = patients[:TRAIN]
val_patients = patients[TRAIN:TRAIN + VAL]
test_patients = patients[TRAIN + VAL:TRAIN + VAL + TEST]

for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST, folder), exist_ok=True)

def copy_cases(cases, folder):
    destination = os.path.join(DEST, folder)

    for i, patient in enumerate(cases, 1):
        src = os.path.join(SOURCE, patient)
        dst = os.path.join(destination, patient)

        if not os.path.exists(dst):
            shutil.copytree(src, dst)

        if i % 50 == 0 or i == len(cases):
            print(folder, i, "/", len(cases))

copy_cases(train_patients, "train")
copy_cases(val_patients, "val")
copy_cases(test_patients, "test")

print()
print("DATA SPLIT COMPLETE")
print("Train:", len(train_patients))
print("Validation:", len(val_patients))
print("Test:", len(test_patients))
print("Location:", DEST)