import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DOWNLOAD = True

# Downlaod datasets
if(DOWNLOAD):
    with open('parquets/hord_diffusiondb_scores.parquet', 'wb') as f:
        f.write(requests.get("https://ratings.aihorde.net/api/v1/download/diffusiondb_export.parquet", allow_redirects=True).content)

    with open('parquets/diffusion_db.parquet', 'wb') as f:
        f.write(requests.get("https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata-large.parquet", allow_redirects=True).content)

diffusion_df = pd.read_parquet('parquets/diffusion_db.parquet')
hord_score_df = pd.read_parquet('parquets/hord_diffusiondb_scores.parquet')

# Filter duplicate ratings from the same user
unique_hord_df = hord_score_df[hord_score_df[["id", "user_id"]].duplicated(keep=False) == False]

# Unique images with different counts:
unique_hord_df = hord_score_df.groupby("id").agg({"ratings_count": "count", "rating": "mean", "artifacts": "mean"}).reset_index()

#filter images with low rating count
unique_hord_df = unique_hord_df[unique_hord_df["ratings_count"] >= 3 ]

diffusion_df = diffusion_df.assign(hord_id=diffusion_df['image_name'].str.split(".", expand=True)[0])
mixed_diffusion_df = pd.merge(left=unique_hord_df, right=diffusion_df, left_on='id', right_on='hord_id')

#Filter images that are scored 2.0 on NSFW as they are just blurred images
mixed_diffusion_df = mixed_diffusion_df[mixed_diffusion_df["image_nsfw"] != 2.0 ]

mixed_diffusion_df = mixed_diffusion_df[["image_name", "ratings_count", "rating", "artifacts", "prompt", "part_id", "width", "height"]]

train_split_df, validate_split_df = np.split(mixed_diffusion_df.sample(frac=1, random_state=42), [int(0.8*len(mixed_diffusion_df))])

mixed_diffusion_df.to_parquet("parquets/prepared_hord_diffusion_dataset.parquet")
train_split_df.to_parquet("parquets/train_split.parquet")
validate_split_df.to_parquet("parquets/validate_split.parquet")


# Set up the figure with two subplots
fig, axs = plt.subplots(3, 2, figsize=(13, 15))

def plot(row, title, df):
     # Plot the first histogram on the first subplot
    axs[row, 0].hist(df["rating"], bins=15, density=False, color='blue', edgecolor='black')
    axs[row, 0].set_xlabel('Rating')
    axs[row, 0].set_ylabel('Count')
    axs[row, 0].set_title(title)

    # Plot the second histogram on the second subplot
    axs[row, 1].hist(df["artifacts"], bins=15, density=False, color='blue', edgecolor='black')
    axs[row, 1].set_xlabel('Artifacts')
    axs[row, 1].set_ylabel('Count')
    axs[row, 1].set_title(title)

plot(0, "Full Dataset", mixed_diffusion_df)
plot(1, "Training Split", train_split_df)
plot(2, "Validation Split", validate_split_df)

# Adjust the layout of the subplots
fig.tight_layout()

fig.savefig('dataset-process/dataset_stats.png')

print(f"Train split of length: {len(train_split_df)}")
print(f"Validate split of length: {len(validate_split_df)}")