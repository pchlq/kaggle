import pandas as pd
from joblib import Parallel, delayed 
import joblib
import glob
from tqdm import tqdm

if __name__ == "__main__":
    files = glob.glob("../input/train_*.parquet")
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop("image_id", axis=1)
        image_array = df.values

        def save_fn(i, img_id):
            joblib.dump(img_array[i, :], f"../input/image_pickles/{img_id}.pkl")

        Parallel(n_jobs=8, backend='multiprocessing')(
            delayed(save_fn)(i, img_id) for i, img_id in tqdm(enumerate(image_ids), total=len(image_ids))
        )