# 1. Download source videos
python data_collection/downloader.py

# 2. Open each video and mark IN/OUT for every sign
python data_collection/annotator.py

# 3. Extract MediaPipe landmarks from every annotation
python data_collection/extract_landmarks.py

# 4. Validate shapes and class balance
python data_collection/verify_dataset.py

# 5. Generate noise/flip/warp variants (3 augmented per original)
python data_collection/augment.py

# 6. Stack into X.npy / y.npy and split into train/val/test
python data_collection/build_dataset.py