"""
Dataset Registry & Ingestion Helper for Deepfake Voice Detection

Contains download endpoints, Zenodo DOIs, and instructions for:
- ASVspoof 2021 DF & LA
- ASVspoof 5 (2024)
- WaveFake
- AI4Bharat IndicSpeech & Shrutilipi (Indian Accents)
- PartialSpoof & ADD
"""

import sys
from typing import Dict, Any

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "asvspoof2021_df": {
        "title": "ASVspoof 2021 Deepfake (DF) Track",
        "url": "https://www.asvspoof.org/index2021.html",
        "zenodo": "https://zenodo.org/record/4835108",
        "description": "Essential benchmark: 100K+ utterances compressed with lossy codecs (G.711, AAC, MP3) simulating telecom transmission.",
        "size": "~25 GB",
        "license": "Research Only"
    },
    "asvspoof5": {
        "title": "ASVspoof 5 (2024 Challenge)",
        "url": "https://www.asvspoof.org/asvspoof5",
        "description": "Latest benchmark featuring modern diffusion TTS, flow-matching, and multi-lingual telephony audio.",
        "size": "~60 GB",
        "license": "Research Only"
    },
    "wavefake": {
        "title": "WaveFake Dataset",
        "url": "https://github.com/joel-frank/wavefake",
        "zenodo": "https://zenodo.org/record/5642694",
        "description": "100,000+ generated audio clips across 6 neural vocoders (MelGAN, HiFi-GAN, WaveGlow, Parallel WaveGAN).",
        "size": "~30 GB",
        "license": "CC-BY 4.0"
    },
    "indic_speech": {
        "title": "AI4Bharat IndicSpeech & Shrutilipi (Indian Languages)",
        "url": "https://ai4bharat.iitm.ac.in/shrutilipi",
        "huggingface": "https://huggingface.co/ai4bharat",
        "description": "6,400+ hours of natural Indian accented speech across 13 regional languages (Hindi, Tamil, Telugu, Marathi, etc.). Crucial for minimizing false alarms in Indian contexts.",
        "size": "~50 GB",
        "license": "CC-BY-NC 4.0"
    },
    "partial_spoof": {
        "title": "PartialSpoof Dataset",
        "url": "https://github.com/piotrkawa/partialspoof",
        "description": "Utterances where only small portions or specific words are spoofed/spliced into genuine human speech.",
        "size": "~10 GB",
        "license": "Research"
    },
    "voxceleb": {
        "title": "VoxCeleb 1 & 2 (Bonafide Real-World Human Speech)",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
        "description": "Over 1M+ utterances from 6,000+ human speakers recorded in noisy, unconstrained real-world environments.",
        "size": "~40 GB",
        "license": "Creative Commons"
    }
}


def print_dataset_guide():
    print("\n" + "=" * 75)
    print(" TRUE TONE: RECOMMENDED DATASET REGISTRY & DOWNLOAD GUIDE")
    print("=" * 75)
    for key, info in DATASET_REGISTRY.items():
        print(f"\n[{key.upper()}] - {info['title']}")
        print(f"  Description: {info['description']}")
        print(f"  URL:         {info['url']}")
        if "zenodo" in info:
            print(f"  Zenodo DOI:  {info['zenodo']}")
        if "huggingface" in info:
            print(f"  HuggingFace: {info['huggingface']}")
        print(f"  Est. Size:   {info['size']} | License: {info['license']}")
    print("\n" + "=" * 75)
    print("To download directly in Python using huggingface_hub or wget:")
    print("  pip install huggingface_hub")
    print("  huggingface-cli download ai4bharat/indic-speech-hindi --local-dir data/indic/")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    print_dataset_guide()
