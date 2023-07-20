import os
import argparse
import demucs.separate
import traceback
import torch
import numpy as np
from pydub import AudioSegment
from my_utils import load_audio
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from fairseq import checkpoint_utils
from config import Config
from vc_infer_pipeline import VC
import scipy.io.wavfile as wavfile

config = Config()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str, help='Model name. Will recursively search models/<name>/ for .pth and .index files')
    parser.add_argument('source_audio_path', type=str, help='Source audio path, e.g., myFolder/MySource.wav.')

    parser.add_argument('--output_filename', type=str, default='MyTest.wav', help="Output file name to be placed in './audio-outputs', e.g., MyTest.wav.")
    parser.add_argument('--feature_index_filepath', type=str, default='logs/mi-test/added_IVF3042_Flat_nprobe_1.index', help="Feature index file path, e.g., logs/mi-test/added_IVF3042_Flat_nprobe_1.index.")
    parser.add_argument('--speaker_id', type=int, default=0, help='Speaker ID, e.g., 0.')
    parser.add_argument('--transposition', type=int, default=0, help='Transposition, e.g., 0.')
    parser.add_argument('--f0_method', type=str, default='harvest', help="F0 method, e.g., 'harvest' (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny).")
    parser.add_argument('--crepe_hop_length', type=int, default=160, help='Crepe hop length, e.g., 160.')
    parser.add_argument('--harvest_median_filter_radius', type=int, default=3, help='Harvest median filter radius (0-7), e.g., 3.')
    parser.add_argument('--post_resample_rate', type=int, default=0, help='Post resample rate, e.g., 0.')
    parser.add_argument('--mix_volume_envelope', type=int, default=1, help='Mix volume envelope, e.g., 1.')
    parser.add_argument('--feature_index_ratio', type=float, default=0.78, help='Feature index ratio (0-1), e.g., 0.78.')
    parser.add_argument('--voiceless_consonant_protection', type=float, default=0.33, help='Voiceless Consonant Protection (Less Artifact). Smaller number = more protection. 0.50 means Do not Use. E.g., 0.33.')

    args = parser.parse_args()
    return args

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["RVCv2/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "RVCv2/weights"
weight_uvr5_root = "RVCv2/uvr5_weights"
index_root = "RVCv2/logs"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


# There can only be one tone in a tab globally
def get_vc(sid, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version

    # Check if sid is empty or not given
    if sid == "" or sid == []:
        global hubert_model

        # If a model exists, it is removed in preparation for a new model
        if hubert_model is not None:
            print("clean_empty_cache")
            # Removing previous model and related variables
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None

            # Clean the cache if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Below is for thorough cleanup
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            # Depending on the version and if_f0, different synthesizer is selected
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

            # Clean up the used variables and cache
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None

        return {"visible": False, "__type__": "update"}

    # Load the model
    # person = "%s/%s" % (weight_root, sid)
    person = sid
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)

    # Set protection levels based on if_f0
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {
            "visible": False,
            "value": 0.5,
            "__type__": "update",
        }
    else:
        to_return_protect0 = {
            "visible": True,
            "value": to_return_protect0,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": True,
            "value": to_return_protect1,
            "__type__": "update",
        }

    # Load the appropriate synthesizer based on version and if_f0
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    # Remove the encoder and load model weights
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))

    # Prepare the model for evaluation and set the type (half or float)
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    # Initialize the voice conversion and set n_spk
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return (
        {"visible": True, "maximum": n_spk, "__type__": "update"},
        to_return_protect0,
        to_return_protect1,
    )



def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
): 
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        ) 
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)
    
def find_pth_and_index_files(directory):
    pth_file = None
    index_file = None

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".pth") and pth_file is None:
                pth_file = os.path.join(root, filename)
            if filename.endswith(".index") and index_file is None:
                index_file = os.path.join(root, filename)
            if pth_file is not None and index_file is not None:
                return pth_file, index_file

    return pth_file, index_file

def separate_track(source_audio_path):
    track_filename = os.path.basename(source_audio_path)
    track_name = os.path.splitext(track_filename)[0]
    # Check if splits already exist to avoid reprocessing
    if os.path.exists(f"separated/htdemucs/{track_name}/vocals.wav") and os.path.exists(f"separated/htdemucs/{track_name}/no_vocals.wav"):
        print("Found pre-separated track, skipping separation.")
        return track_name
    # Separate the track with demucs
    # Written to /seperated/htdemucs/{track_name}/vocals.wav
    demucs.separate.main(["--two-stems", "vocals", source_audio_path])
    return track_name

def join_track(track_name, model_name):
    # Load the audio files
    vocal = AudioSegment.from_wav(f"RVCv2/audio-outputs/{track_name}_{model_name}_vocals.wav")
    instrumental = AudioSegment.from_wav(f"separated/htdemucs/{track_name}/no_vocals.wav")

    # Combine the audio files
    combined = vocal.overlay(instrumental)

    return combined


def main():
    args = parse_args()
    print("Searching for model...")
    pth_file, index_file = find_pth_and_index_files(f"models/{args.model_name}/")
    if pth_file is None:
        print("No model file found.")
        return
    print("Found model file: %s" % pth_file)
    print("Found index file: %s" % index_file)
    print("---------------------------------")
    print("Demucs: Starting track separation...")
    track_name = separate_track(args.source_audio_path)
    print("Demucs: Track separation complete.")
    print("---------------------------------")
    print("RVCv2: Starting the inference...")
    vc_data = get_vc(pth_file, {}, {})
    print(vc_data)
    print("RVCv2: Performing inference...")
    conversion_data = vc_single(
        args.speaker_id,
        f"separated/htdemucs/{track_name}/vocals.wav",
        args.transposition,
        None,
        args.f0_method,
        index_file if index_file is not None else "",
        index_file if index_file is not None else "",
        args.feature_index_ratio,
        args.harvest_median_filter_radius,
        args.post_resample_rate,
        args.mix_volume_envelope,
        args.voiceless_consonant_protection,
        args.crepe_hop_length,        
    )
    if "Success." in conversion_data[0]:
        print("RVCv2: Inference succeeded. Writing to %s/%s..." % ('RVCv2/audio-outputs', f"{track_name}_{args.model_name}_vocals.wav"))
        wavfile.write('%s/%s' % ('RVCv2/audio-outputs', f"{track_name}_{args.model_name}_vocals.wav"), conversion_data[1][0], conversion_data[1][1])
        print("RVCv2: Finished! Saved output to %s/%s" % ('RVCv2/audio-outputs', f"{track_name}_{args.model_name}_vocals.wav"))
        print("---------------------------------")
        print("Rejoing the track...")
        joined_track = join_track(track_name, args.model_name)
        print("Track rejoined.")
        print("Writing completed file...")
        joined_track.export(f"RVCv2/audio-outputs/{track_name}_{args.model_name}.wav", format='wav')
        print("Track successfully written to: " + f"RVCv2/audio-outputs/{track_name}_{args.model_name}.wav")
        print("Cleaning up vocal track...")
        os.remove(f"RVCv2/audio-outputs/{track_name}_{args.model_name}_vocals.wav")
        print("---------------------------------")
        print("Inference complete.")
    else:
        print("RVCv2: Inference failed. Here's the traceback: ")
        print(conversion_data[0])


if __name__ == '__main__':
    main()
