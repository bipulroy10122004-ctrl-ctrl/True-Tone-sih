import os
import soundfile as sf

meta_path = r"D:\True Tone SIH\keys_2021_DF\keys\DF\CM\trial_metadata.txt"
flac_dir = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac"

# Load bonafide trial IDs
bonafide_tids = set()
with open(meta_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 6 and parts[5] == "bonafide":
            bonafide_tids.add(parts[1])

print(f"Total bonafide in metadata: {len(bonafide_tids)}")

existing_flacs = set(os.listdir(flac_dir))
existing_bonafide = [tid for tid in bonafide_tids if f"{tid}.flac" in existing_flacs]
print(f"Existing bonafide flac files on disk: {len(existing_bonafide)}")

valid_bonafide = []
invalid_bonafide = []

for tid in existing_bonafide:
    fpath = os.path.join(flac_dir, f"{tid}.flac")
    try:
        data, sr = sf.read(fpath)
        if len(data) > 0:
            valid_bonafide.append(tid)
    except Exception:
        invalid_bonafide.append(tid)

print(f"Valid & readable bonafide files: {len(valid_bonafide)} ({len(valid_bonafide)/len(existing_bonafide)*100:.1f}%)")
print(f"Corrupted/truncated bonafide files: {len(invalid_bonafide)}")
