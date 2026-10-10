import tarfile
import os

p0 = r"D:\True Tone SIH\ASVspoof2021_DF_eval_part00.tar.gz"
p1 = r"D:\True Tone SIH\ASVspoof2021_DF_eval_part01.tar.gz"
p2 = r"D:\True Tone SIH\ASVspoof2021_DF_eval_part02.tar.gz"

for p in [p0, p1, p2]:
    print(f"Checking {os.path.basename(p)}...")
    try:
        with tarfile.open(p, "r:gz") as tar:
            members = tar.getmembers()[:3]
            print(f"  Valid tar.gz! First 3 members: {[m.name for m in members]}")
    except Exception as e:
        print(f"  Failed: {e}")
