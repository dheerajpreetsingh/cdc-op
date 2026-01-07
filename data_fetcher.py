import os
import time
import requests
import pandas as pd

MAPBOX_TOKEN = "your_token"

ZOOM = 17.5
WIDTH = 400
HEIGHT = 400

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TRAIN_IMG_DIR = "train_img"
TEST_IMG_DIR = "test_img"


def load_bad_ids(file):
    if os.path.exists(file):
        return set(pd.read_csv(file)["id"].astype(str))
    return set()


# load bad id lists (if they exist)
BAD_TRAIN = load_bad_ids("bad_train_image_ids.csv")
BAD_TEST  = load_bad_ids("bad_test_image_ids.csv")

def get_lat_lon(row):
    lat = row["lat"] if pd.notna(row["lat"]) else None
    lon = row["long"] if pd.notna(row["long"]) else None
    return lat, lon

def fetch_and_save_images(df, out_dir, name="dataset"):
    os.makedirs(out_dir, exist_ok=True)

    bad_ids = BAD_TRAIN if name == "train" else BAD_TEST

    for _, row in df.iterrows():
        prop_id = str(row["id"])
        save_path = os.path.join(out_dir, f"{prop_id}.png")

        lat, lon = get_lat_lon(row)
        if lat is None or lon is None:
            print(f"[SKIP] {name} id={prop_id} — missing lat/lon")
            continue

        # ---- DECISION LOGIC (preserves original behavior) ----
        if os.path.exists(save_path):
            if prop_id in bad_ids:
                print(f"[REDOWNLOAD] {name} id={prop_id} — marked bad, overwriting…")
            else:
                print(f"[SKIP] {name} id={prop_id} — already downloaded & valid")
                continue
        # ------------------------------------------------------

        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{lon},{lat},{ZOOM}/{WIDTH}x{HEIGHT}"
            f"?access_token={MAPBOX_TOKEN}"
        )

        try:
            r = requests.get(url, timeout=15)

            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                print(f"[OK] {name} id={prop_id} — saved")
            else:
                print(f"[ERROR] id={prop_id} status={r.status_code}")

        except Exception as e:
            print(f"[FAILED] id={prop_id} error={e}")

        time.sleep(0.05)

def main():
    print("Loading datasets...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print("Downloading TRAIN images...")
    fetch_and_save_images(train, TRAIN_IMG_DIR, name="train")

    print("Downloading TEST images...")
    fetch_and_save_images(test, TEST_IMG_DIR, name="test")

    print("Done!")

if __name__ == "__main__":
    main()
