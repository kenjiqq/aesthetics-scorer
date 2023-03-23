from img2dataset import download
import shutil
import os

output_dir = os.path.abspath("laion2b-en/dataset")

if __name__ == '__main__':
    download(
        processes_count=8,
        thread_count=16,
        url_list="laion2b-en/parquet/dataset.parquet",
        image_size=256,
        output_folder=output_dir,
        output_format="webdataset",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=10000,
        distributor="multiprocessing",
    )