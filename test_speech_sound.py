import os
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.set_device(0)
print("VISIBLE:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_VISIBLE inside container:", torch.cuda.device_count())
import torchaudio
import toml
import json
import random
import utils
import diffusion
import numpy as np
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from loss_metric import _calculate_sisnr, pit_sisnr
import models

seed = 17 
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  



def load_data_list(info_json_path, label_json_path):
    # read data info file
    with open(info_json_path, 'r', encoding='utf-8') as f:
        data_info = json.load(f)
        
    # read label
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    # load data infos
    items = []
    for info in data_info:
        items.append({
            'length': info.get('wave_length'), 
            'wav_path': info.get('wave_path'),
            'label':    info.get('wave_label'), 
            'spans':    info.get('spans'),
            'span_probs': info.get('span_probs')
        })

    # category nums
    num_categories = len(label_map)
    ordered_groups = [[] for _ in range(num_categories)]
    
    for item in items:
        string_label = item.get('label')
        if string_label in label_map:
            index = label_map[string_label] 
        ordered_groups[index].append(item)
        
    return ordered_groups, label_map


def _get_VCTK_paths(root_dir, used_spk_num, test = True):
    spk_paths = []
    speech_paths = []
    i = 0
    for spk in sorted(os.listdir(root_dir)):
        spk_dir = os.path.join(root_dir, spk)
        if os.path.isdir(spk_dir):
            spk_utterances = []
            for f in os.listdir(spk_dir):
                if f.endswith('.flac'):
                    audio_path = os.path.join(spk_dir, f)
                    spk_utterances.append(audio_path)
                    speech_paths.append(audio_path)
            spk_paths.append(spk_utterances)
        i += 1
    
    len_data = len(spk_paths) 
    random.shuffle(spk_paths)
    if test:
        spk_paths = spk_paths[used_spk_num:]
    else:
        spk_paths = spk_paths[:used_spk_num] 
    print('num spk', len(spk_paths))
    print('num speech', len(speech_paths))
    return spk_paths, speech_paths


def make_data(ordered_groups, label_map, 
              speech_data,
              n_src, file_sr, tgt_sr, tgt_len_s):
    # length
    target_frames = int(tgt_len_s * tgt_sr)

    # all audio
    loaded_audios = []
    total_label = []
    
    # 1. choose speech, vctk 44100Hz
    choosen_speech_path, spk_id= random.choice(speech_data) 
    audio, sr = torchaudio.load(choosen_speech_path)
    if sr != tgt_sr:
        audio = torchaudio.transforms.Resample(sr, tgt_sr)(audio)
    if audio.shape[-1] > target_frames:
        start = random.randint(0, audio.shape[-1] - target_frames)
        audio = audio[..., start : start + target_frames]
    loaded_audios.append(audio)
    total_label.append(0)

    # 2. load sound event
    choosen_src_indices = random.sample(range(len(ordered_groups)), n_src)
    used_srcs = [ordered_groups[i] for i in choosen_src_indices]
    for src_info in used_srcs:
        used_src_info = random.choice(src_info)
        path = used_src_info.get('wav_path')
        spans = used_src_info.get('spans')
        span_probs = used_src_info.get('span_probs')
        label = label_map[used_src_info.get('label')]
        
        chosen_span_times = random.choices(population=spans, weights=span_probs, k=1)[0]
        t0_sec, t1_sec = chosen_span_times[0], chosen_span_times[1]
        
        start_frame_orig = int(t0_sec * file_sr)
        num_frames_orig_to_load = int((t1_sec - t0_sec) * file_sr)
        
        audio, orig_sr = torchaudio.load(
            path, 
            frame_offset=start_frame_orig, 
            num_frames=num_frames_orig_to_load
        )
        
        if orig_sr != tgt_sr:
            audio = torchaudio.transforms.Resample(orig_sr, tgt_sr)(audio)
        
        if audio.shape[-1] > target_frames:
            start = random.randint(0, audio.shape[-1] - target_frames)
            audio = audio[..., start : start + target_frames]
        
        loaded_audios.append(audio)
        total_label.append(label)

    # non silence overlap
    total_src_on_canvas = []
    last_placement_info = {'start': -1, 'end': -1}

    for i, audio in enumerate(loaded_audios):
        current_frames = audio.shape[-1]
        canvas = torch.zeros(1, target_frames, device=audio.device)

        if i == 0:
            start_frame = random.randint(0, target_frames - current_frames)
        else:
            prev_start = last_placement_info['start']
            prev_end = last_placement_info['end']
            min_start = prev_start
            max_start = min(prev_end, target_frames - current_frames)

            if min_start >= max_start:
                start_frame = min(prev_start, target_frames - current_frames)
            else:
                start_frame = random.randint(min_start, max_start)

        canvas[..., start_frame : start_frame + current_frames] = audio
        last_placement_info['start'] = start_frame
        last_placement_info['end'] = start_frame + current_frames
        
        # energy
        canvas, _ = change_energy(canvas)
        total_src_on_canvas.append(canvas.squeeze(0))
    
    # mix all sources
    single_sources = torch.stack(total_src_on_canvas, dim=0) # Shape: [n_src, T]
    mixture = torch.sum(single_sources, dim=0) # Shape: [T]
        
    return mixture, single_sources, total_label


def adjust_wave_to_tgt_len(
    waveform_tensor: torch.Tensor, 
    target_frames: int):
    num_channels, current_frames = waveform_tensor.shape
    
    operation_details = {} 
    adjusted_waveform = torch.zeros((num_channels, target_frames), 
                                    device=waveform_tensor.device, 
                                    dtype=waveform_tensor.dtype)

    if current_frames < target_frames:
        operation_details['operation'] = 'pad'
        pad_needed_frames = target_frames - current_frames
        pad_start_frames = random.randint(0, pad_needed_frames)
        operation_details['pad_start_frames'] = pad_start_frames
        start_idx_in_target = pad_start_frames
        end_idx_in_target = pad_start_frames + current_frames
        adjusted_waveform[:, start_idx_in_target:end_idx_in_target] = waveform_tensor
    elif current_frames > target_frames:
        operation_details['operation'] = 'crop'
        over_length_frames = current_frames - target_frames
        crop_start_frame_in_source = random.randint(0, over_length_frames)
        operation_details['crop_start_frame'] = crop_start_frame_in_source
        adjusted_waveform = waveform_tensor[:, crop_start_frame_in_source : crop_start_frame_in_source + target_frames]
    else: 
        operation_details['operation'] = 'none'
        adjusted_waveform = waveform_tensor.clone() 

    return adjusted_waveform, operation_details


def change_energy(audio):
    current_rms = torch.sqrt(torch.mean(audio**2))
    current_rms_db = 20 * torch.log10(current_rms)
    
    # tgt db
    tgt_db = round(random.uniform(-25, -20), 3)
    
    # db diff
    db_difference = tgt_db - current_rms_db 
    amplify_ratio = 10 ** (db_difference / 20)
    audio = audio * amplify_ratio
    
    # info
    info = {'amplify_ratio':amplify_ratio, 'tgt_db':tgt_db}
    
    return audio, info


def degradation(x: torch.Tensor, n_src = 2) -> torch.Tensor:
    min_sample_length = x.shape[-1] // n_src
    segments = torch.split(x, min_sample_length, dim=-1) 
    degraded = sum(segments)
    return degraded


def _norm(audio, eps: float = 1e-8):
    peak = audio.abs().amax(dim=-1, keepdim=True)
    peak = peak.clamp(min=eps)
    audio = audio / peak
    audio = audio * 0.95
    return audio, 0.95 / peak


def evaluate(model, diffusion, device, ordered_groups, label_map, 
             speech_data, save_root, n_src, sample_num):
    file_sr = 44100 # Hz
    tgt_sr = 16000 # Hz
    tgt_len_sec = 4 # 4 s    
    total_snr = []
    mean_snr = []
    all_mean_snr = []
    
    # save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_root, timestamp)
    os.makedirs(save_path, exist_ok=True)
    
    # sample num
    sample_num = sample_num
    all_snr = []
    for item in tqdm(range(sample_num), total=sample_num):
        # make data, mixture
        file_sr = 44100 # Hz
        tgt_sr = 16000 # Hz
        tgt_len_sec = 4 # 4 s
        n_src = n_src - 1 # 1 speech + n sound
        mixture, single_sources, total_label = make_data(ordered_groups, 
                                                         label_map, 
                                                         speech_data, 
                                                         n_src, 
                                                         file_sr, # fsd sr
                                                         tgt_sr, 
                                                         tgt_len_sec)
        
        # norm
        mixture, peak = _norm(mixture)
        mixture = mixture.to(device)
        mixture = mixture.reshape(1, 1, -1)

        # model label
        n_src_num = n_src + 1 
        model_kwargs_in = torch.tensor(total_label).to(device)
        
        generator = diffusion.p_sample_loop_group(model,
                                            shape = single_sources.reshape(1, 1, -1).shape,
                                            measurement = mixture,
                                            orig_x = single_sources.reshape(1, 1, -1).to(device),
                                            n_src = n_src_num,
                                            clip_denoised = True,
                                            degradation = degradation,
                                            model_kwargs = model_kwargs_in) 

        # direct sample
        # generator = diffusion.direct_sample_group(model,
        #                                         shape = mixture.shape,
        #                                         measurement = mixture,
        #                                         n_src = n_src_num,
        #                                         clip_denoised = True,
        #                                         degradation = degradation,
        #                                         model_kwargs = model_kwargs_in)



        final_sample = None
        for out in generator:  
            final_sample = out["sample"]
        sepa_srcs = final_sample.cpu() 
        sepa_srcs_chunked = torch.chunk(
            sepa_srcs, chunks = n_src_num, dim=-1
        )

        # metric
        estimated_sources = torch.stack([sepa_srcs_chunked[i].squeeze(0) for i in range(n_src_num)])
        estimated_sources = estimated_sources.squeeze(1)
        reference_sources = torch.stack([(single_sources[i] * peak).squeeze(0) for i in range(n_src_num)])

        # perm SI-SNR
        avg_sisnr, best_perm = pit_sisnr(estimated_sources, reference_sources)
        
        # save wav
        single_mean_snr = 0
        for i in range(n_src_num):
            best_idx = best_perm[i]
            estimated_source = estimated_sources[best_idx]
            reference_source = reference_sources[i]
            
            sisnr_value = _calculate_sisnr(estimated_source, reference_source)
            sisnr_value = float(sisnr_value)
            total_snr.append(sisnr_value)
            single_mean_snr += sisnr_value
            
            # save audio
            save_path_root = os.path.join(save_path, str(item))
            os.makedirs(save_path_root, exist_ok=True)
            save_wave_path = os.path.join(save_path_root, f"sepa_{i}.wav")
            torchaudio.save(save_wave_path, estimated_source.unsqueeze(0).to('cpu'), tgt_sr)
            save_wave_path = os.path.join(save_path_root, f"real_{i}.wav")
            torchaudio.save(save_wave_path, reference_source.unsqueeze(0).to('cpu'), tgt_sr)
            
        save_wave_path = os.path.join(save_path_root, f"mix.wav")
        mixture_save = mixture.squeeze(0).cpu() 
        torchaudio.save(save_wave_path, mixture_save, tgt_sr) 
        mean_snr.append(single_mean_snr / n_src_num)
        all_snr.append(single_mean_snr / n_src_num)
        all_snr_mean = sum(all_snr) / len(all_snr)
        snr_filename = os.path.join(save_path, "si_snr_results.json")
        current_result = {
            'all_snr_mean': all_snr_mean,
            "total_snr": total_snr,
            "mean_snr": mean_snr
        }
        with open(snr_filename, 'w') as f:
            json.dump(current_result, f, indent=4) 

    print(f"Saved SI-SNR results to {snr_filename}")
            

def load_model(config, device):
    model_name = config["model_name"]
    model_class = getattr(models, model_name)
    model = model_class(config["model_cfg"])
    for param in model.parameters():
        param.requires_grad = False

    # paras info
    print('model:',model_name, 'paras num:', utils.get_paras_num(model))
    print('ckpt path:', config["ckpt_path"])
    
    # load state_dict
    ckpt = torch.load(config["ckpt_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # ema
    ema_model = deepcopy(model)
    ema_model.load_state_dict(ckpt["ema"])
    ema_model.to(device).eval()
    
    return ema_model


def load_one_model(config_path, device):
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f) if ext == '.json' else toml.load(f)
    return load_model(config, device)


def main(device):
    # 1. save path
    save_root = './test_results/sound_speech_test'
        
    # 2. not full version of VCTK dataset, load speech list
    speech_dir = './audio_samples/speech'
    spk_paths, speech_paths = _get_VCTK_paths(speech_dir, used_spk_num = 100, test = True)
    speech_data = [] 
    for spk_id, utterances in enumerate(spk_paths):
        for u in utterances:
            speech_data.append((u, spk_id))

    # 3. not full version of FSD dataset, load sound list
    ordered_groups, label_map = load_data_list(
        info_json_path = './sound_dataset_process/audio_infos/test.json', # audio paths 
        label_json_path = './sound_dataset_process/audio_infos/label.json' # audio label info
        )
    
    # 4. modify this, load speech model config file 
    speech_config_path = './config/atten_unet_vctk/config.toml'
    speech_model = load_one_model(speech_config_path, device)
    
    # 5. modify this, load sound model config file 
    sound_config_paths = [
    './config/atten_unet_fsd/config.toml',
    ] 
    sound_models = [load_one_model(p, device) for p in sound_config_paths]

    # total model
    total_model = [speech_model, *sound_models]
    
    # diffusion
    ext = os.path.splitext(speech_config_path)[1].lower()
    with open(speech_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f) if ext == '.json' else toml.load(f)
    gaus_diffusion = diffusion.GaussianDiffusion(
                                steps = 200, 
                                config_file = config,
                                beta_start = config['train_para']['beta_start'],
                                beta_end = config['train_para']['beta_end'])
    
    # run
    evaluate(total_model, gaus_diffusion, device, 
             ordered_groups = ordered_groups, label_map = label_map,
             speech_data = speech_data,
             save_root = save_root,
             n_src = 2, # 1 speech + n sound
             sample_num = 2)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    main(device)

    