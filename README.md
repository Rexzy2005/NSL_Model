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


android: 527745842570-i8u61c1ehtct9bion3gcd55rshsqnldv.apps.googleusercontent.com

web: 527745842570-g7gsms10aaksoakfsbtduvdph1hf82ua.apps.googleusercontent.com

Now i have gotten my: Project url: https://aosbjpbtdnwmozopuqgv.supabase.co, my publishabe url: sb_publishable_tGxsQgC30bKKCt2wHJM22g_Tz1Hh7iX, postgresql://postgres:NSL_Translator%402026@db.aosbjpbtdnwmozopuqgv.supabase.co:5432/postgres

C:\flutter\bin\flutter.bat run `
  --dart-define=SUPABASE_URL=https://aosbjpbtdnwmozopuqgv.supabase.co `
  --dart-define=SUPABASE_PUBLISHABLE_KEY=sb_publishable_tGxsQgC30bKKCt2wHJM22g_Tz1Hh7iX



