# Based on from https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/visulaize_100k_from_LAION400M.py

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from aesthetics_scorer.model import load_model, preprocess

CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
RATING_MODEL = "aesthetics_scorer/models/aesthetics_scorer_rating_openclip_vit_h_14.pth"
ARTIFACTS_MODEL = "aesthetics_scorer/models/aesthetics_scorer_artifacts_openclip_vit_h_14.pth"

laion_embedding_df = pd.read_parquet("parquets/laion_embeddings.parquet")
laion_info = pd.read_parquet("laion2b/parquet/dataset.parquet")

#merge laion embeddings with test on image_url and URL
laion_embedding_df = laion_embedding_df.merge(laion_info, left_on="image_url", right_on="URL")

# filter nsfw images
laion_embedding_df = laion_embedding_df[(laion_embedding_df["NSFW"] != "UNSURE") & (laion_embedding_df["NSFW"] != "NSFW")]

device = "cuda" if torch.cuda.is_available() else "cpu"

rating_model = load_model(RATING_MODEL).to(device)
rating_model.eval()

artifacts_model = load_model(ARTIFACTS_MODEL).to(device)
artifacts_model.eval()

embeddings = np.array(laion_embedding_df["pooled_output"].to_list())


urls = np.array(laion_embedding_df["image_url"].to_list())
rating_predictions = []
artifacts_predictions = []

BATCH_SIZE = 1024
for pos in tqdm(range(0, len(embeddings), BATCH_SIZE)):
   batch_embeddings = embeddings[pos:pos+BATCH_SIZE]
   batch_embeddings = preprocess(torch.Tensor(batch_embeddings).to(device))
   with torch.no_grad():
      rating_prediction = torch.clamp(rating_model(batch_embeddings), 1, 10)
      artifacts_prediction  = torch.clamp(artifacts_model(batch_embeddings), 0, 5)
   rating_predictions.extend([x.item() for x in rating_prediction.detach().cpu()])
   artifacts_predictions.extend([x.item() for x in artifacts_prediction.detach().cpu()])


for predictions, name, min_value, max_value in ((rating_predictions, "rating", 1, 10), (artifacts_predictions, "artifacts", 0, 5)):
    df = pd.DataFrame(list(zip(urls, predictions)),
    columns =['filepath', 'prediction'])


    BUCKET_STEP = 0.5
    buckets = [(i, i + BUCKET_STEP) for i in np.arange(min_value, max_value, BUCKET_STEP)]


    html= f"<h1>Aesthetic {name} scores for {len(predictions)} LAION 5b samples</h1>"

    for [a,b] in buckets:
        total_part = df[(  (df["prediction"] ) *1>= a) & (  (df["prediction"] ) *1 <= b)]
        count_part = len(total_part) / len(df) * 100
        estimated =int ( len(total_part) )
        part = total_part.sample(min(len(total_part), 50))

        html+=f"<h2>In bucket {a} - {b} there is {count_part:.2f}% samples:{estimated:.2f} </h2> <div>"
        for filepath in part["filepath"]:
            html+='<img src="'+filepath +'" height="200" />'

        html+="</div>"
    with open(f"visualize/laion5b-{name}-visualize.html", "w") as f:
        f.write(html)
    
