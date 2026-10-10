import os
from collections import Counter

meta_path = r"D:\True Tone SIH\keys_2021_DF\keys\DF\CM\trial_metadata.txt"
flac_dir = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac"

# Load trial labels from metadata
trial_map = {}
with open(meta_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 6:
            trial_id = parts[1]
            label = parts[5] # 'spoof' or 'bonafide'
            codec = parts[2]
            source = parts[3]
            attack = parts[4]
            trial_map[trial_id] = (label, codec, source, attack)

print(f"Loaded {len(trial_map)} metadata entries.")

# Scan existing flac files
existing_flacs = os.listdir(flac_dir)
print(f"Found {len(existing_flacs)} flac files on disk.")

matched_counts = Counter()
matched_sources = Counter()
matched_codecs = Counter()
matched_attacks = Counter()

bonafide_files = []
spoof_files = []

for fname in existing_flacs:
    if fname.endswith(".flac"):
        tid = fname[:-5]
        if tid in trial_map:
            label, codec, source, attack = trial_map[tid]
            matched_counts[label] += 1
            matched_sources[source] += 1
            matched_codecs[codec] += 1
            matched_attacks[attack] += 1
            if label == "bonafide":
                bonafide_files.append(fname)
            else:
                spoof_files.append(fname)

print(f"Matched labels in existing flac folder:")
for k, v in matched_counts.items():
    print(f"  {k}: {v}")

print(f"\nTop 5 Sources: {matched_sources.most_common(5)}")
print(f"Top 5 Codecs: {matched_codecs.most_common(5)}")
print(f"Top 5 Attack types: {matched_attacks.most_common(5)}")
