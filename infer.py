import platform
import os
import argparse
import traceback
import torch
import numpy as np
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
    parser = argparse.ArgumentParser(description='Voice Inference Script')
    
    parser.add_argument('model_name', type=str, help='Model name with .pth in ./weights, e.g., mi-test.pth.')
    parser.add_argument('source_audio_path', type=str, help='Source audio path, e.g., myFolder/MySource.wav.')
    parser.add_argument('output_filename', type=str, help="Output file name to be placed in './audio-outputs', e.g., MyTest.wav.")
    parser.add_argument('feature_index_filepath', type=str, help="Feature index file path, e.g., logs/mi-test/added_IVF3042_Flat_nprobe_1.index.")
    parser.add_argument('speaker_id', type=int, help='Speaker ID, e.g., 0.')
    parser.add_argument('transposition', type=int, help='Transposition, e.g., 0.')
    parser.add_argument('f0_method', type=str, help="F0 method, e.g., 'harvest' (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny).")
    parser.add_argument('crepe_hop_length', type=int, help='Crepe hop length, e.g., 160.')
    parser.add_argument('harvest_median_filter_radius', type=int, help='Harvest median filter radius (0-7), e.g., 3.')
    parser.add_argument('post_resample_rate', type=int, help='Post resample rate, e.g., 0.')
    parser.add_argument('mix_volume_envelope', type=int, help='Mix volume envelope, e.g., 1.')
    parser.add_argument('feature_index_ratio', type=float, help='Feature index ratio (0-1), e.g., 0.78.')
    parser.add_argument('voiceless_consonant_protection', type=float, help='Voiceless Consonant Protection (Less Artifact). Smaller number = more protection. 0.50 means Do not Use. E.g., 0.33.')

    return parser.parse_args()

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
            version
            # protect,
            # crepe_hop_length
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

def set_env_var_for_mac():
    if platform.system() == 'Darwin':  # Darwin indicates it's a Mac
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("Environment variable set: PYTORCH_ENABLE_MPS_FALLBACK=1")


def main():
    args = parse_args()
    print("RVCv2: Starting the inference...")
    set_env_var_for_mac()
    vc_data = get_vc(args.model_name, {}, {})
    print(vc_data)
    print("RVCv2: Performing inference...")
    conversion_data = vc_single(
        args.speaker_id,
        args.source_audio_path,
        args.transposition,
        None,
        args.f0_method,
        args.feature_index_filepath,
        args.feature_index_filepath,
        args.feature_index_ratio,
        args.harvest_median_filter_radius,
        args.post_resample_rate,
        args.mix_volume_envelope,
        args.voiceless_consonant_protection,
        args.crepe_hop_length,        
    )
    if "Success." in conversion_data[0]:
        print("RVCv2: Inference succeeded. Writing to %s/%s..." % ('RVCv2/', args.output_filename))
        wavfile.write('%s/%s' % ('RVCv2/', args.output_filename), conversion_data[1][0], conversion_data[1][1])
        print("RVCv2: Finished! Saved output to %s/%s" % ('', args.output_filename))
    else:
        print("RVCv2: Inference failed. Here's the traceback: ")
        print(conversion_data[0])


if __name__ == '__main__':
    main()
