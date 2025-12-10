import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import resampy
from tqdm import tqdm 

# this is used to preprocess FSDKaggle2018 dataset
# modify this to generate sound json file
CSV_PATH      = '/misc/export3/corpus/FSDKaggle2018/meta/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv' # from downloaded dataset
AUDIO_DIR     = '/misc/export3/corpus/FSDKaggle2018/audio_test/FSDKaggle2018.audio_test'
OUT_JSON      = '/misc/export3/corpus/FSDKaggle2018/audio_infos/test.json'

# CSV_PATH      = '/misc/export3/corpus/FSDKaggle2018/meta/FSDKaggle2018.meta/train_post_competition.csv' # from downloaded dataset
# AUDIO_DIR     = '/misc/export3/corpus/FSDKaggle2018/audio_train/FSDKaggle2018.audio_train'
# OUT_JSON      = '/misc/export3/corpus/FSDKaggle2018/audio_infos/train_info.json'

SR            = 44100        
WIN_SEC       = 0.05           
HOP_SEC       = 0.025         
ENERGY_THRESH = 1e-4         

def compute_energy(y, sr=SR, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    win   = int(win_sec * sr)
    hop   = int(hop_sec * sr)
    sq    = y**2
    frames= (len(y) - win)//hop + 1
    return np.array([sq[i*hop:i*hop+win].sum() for i in range(frames)])

def energy_to_spans(energy, win_sec=WIN_SEC, hop_sec=HOP_SEC, thresh=ENERGY_THRESH):
    spans, start = [], None
    for i, e in enumerate(energy):
        t0, t1 = i*hop_sec, i*hop_sec + win_sec
        if e > thresh:
            if start is None:
                start = t0
        else:
            if start is not None:
                spans.append([round(start,3), round(t1-1e-6,3)])
                start = None
    if start is not None:
        spans.append([round(start,3), round((len(energy)-1)*hop_sec + win_sec,3)])
    return spans

def main():
    df = pd.read_csv(CSV_PATH)
    records = df.values.reshape(-1, 5)
    records = pd.DataFrame(records, columns=[
        'filename', 'label', 'license', 'duration_ms', 'license_type'
    ])

    output = []
    for _, row in tqdm(records.iterrows(), total=len(records), desc="Processing rows"):
        fname = row['filename']
        label = row['label']
        wav_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.isfile(wav_path):
            print(f"Warning: not found {wav_path}")
            continue

        y, fs = sf.read(wav_path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if fs != SR:
            y = resampy.resample(y, fs, SR)

        energy = compute_energy(y)
        spans  = energy_to_spans(energy)

        lengths = [end - start for start, end in spans]
        total_len = sum(lengths)
        span_probs = [l / total_len for l in lengths]

        if spans:
            output.append({
                'wave_path':   wav_path,
                'wave_label':  label,
                'spans':       spans,
                'span_probs':  span_probs,
                'wave_length': round(len(y) / SR, 3)
            })

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done: exported {len(output)} entries to {OUT_JSON}")

if __name__ == '__main__':
    main()