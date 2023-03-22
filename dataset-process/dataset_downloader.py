import os
import zipfile
import utils.download as download

DATASET_PATH = "W:/diffusiondb/"

for index in range(1, 420 + 1): # dataset is in parts 1->420
    if index % 10 == 0:
        print(f"Progress: {index}")
    part_path  = os.path.join(DATASET_PATH, f"part-{'{:06d}'.format(index)}.zip")
    if os.path.exists(part_path):
        with zipfile.ZipFile(part_path) as z:
            if len(z.filelist) != 1001:
                print(f"invalid filecount in part {index}, found {len(z.filelist)}, will download...")
            else:
                continue
    print(f"missing part {index}, downloading...")
    download.main(index, index+1, DATASET_PATH, large=True)