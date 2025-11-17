import lmdb
import os
from PIL import Image
import numpy as np
import tqdm

dataset_dir = '/home/ju/Desktop/Document_Binarization/Testset'
image_dir = os.path.join(dataset_dir, 'image')
label_dir = os.path.join(dataset_dir, 'mask')

lmdb_dir = '/home/ju/Desktop/lmdb/Testset'
os.makedirs(lmdb_dir, exist_ok=True)

map_size = int(1e11) 
batch_commit = 1000  

env = lmdb.open(lmdb_dir, map_size=map_size)
txn = env.begin(write=True)
count = 0

image_files = sorted(os.listdir(image_dir))

for img_name in tqdm.tqdm(image_files, desc='Processing images'):
    img_path = os.path.join(image_dir, img_name)

    img_name_no_ext = os.path.splitext(img_name)[0]
    label_path = os.path.join(label_dir, img_name_no_ext + '.png')

    if not os.path.exists(label_path):
        print(f"Warning: label not found for {img_name}, skipping.")
        continue

    img = Image.open(img_path).convert('RGB')
    label = Image.open(label_path).convert('L') 

    img_arr = np.array(img, dtype=np.uint8)
    label_arr = np.array(label, dtype=np.uint8)

    img_bytes = img_arr.tobytes()
    label_bytes = label_arr.tobytes()

    key = f'{count:08d}'

    txn.put(f'{key}_image'.encode('ascii'), img_bytes)
    txn.put(f'{key}_label'.encode('ascii'), label_bytes)
    count += 1

    if count % batch_commit == 0:
        txn.commit()
        txn = env.begin(write=True)

txn.commit()
env.close()
print(f"Done! {count} images stored in LMDB at {lmdb_dir}")