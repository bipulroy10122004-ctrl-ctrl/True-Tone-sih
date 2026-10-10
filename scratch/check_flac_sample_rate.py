import os
import soundfile as sf
from tqdm import tqdm

flac_dir = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac"
files = [os.path.join(flac_dir, f) for f in os.listdir(flac_dir)[:2000]]

success = 0
failed = 0
errors = {}

for f in files:
    try:
        data, sr = sf.read(f)
        success += 1
    except Exception as e:
        failed += 1
        err_msg = str(e)
        errors[err_msg] = errors.get(err_msg, 0) + 1

print(f"Tested {len(files)} files:")
print(f"  Success: {success} ({success/len(files)*100:.1f}%)")
print(f"  Failed: {failed} ({failed/len(files)*100:.1f}%)")
print("Top errors:")
for k, v in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  [{v}x] {k}")
