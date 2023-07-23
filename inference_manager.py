import os
import re
import argparse
import demucs.separate
import yt_dlp
import traceback
import torch
import numpy as np
from pydub import AudioSegment
from .my_utils import load_audio
from .infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from fairseq import checkpoint_utils
from .config import Config
from .vc_infer_pipeline import VC
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
config = Config()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str,
                        help='Model name. Will recursively search models/<name>/ for .pth and .index files')
    parser.add_argument('source_audio_path', type=str, help='Source audio path, e.g., myFolder/MySource.wav.')

    parser.add_argument('--output_filename', type=str, default='MyTest.wav',
                        help="Output file name to be placed in './audio-outputs', e.g., MyTest.wav.")
    parser.add_argument('--feature_index_filepath', type=str, default='logs/mi-test/added_IVF3042_Flat_nprobe_1.index',
                        help="Feature index file path, e.g., logs/mi-test/added_IVF3042_Flat_nprobe_1.index.")
    parser.add_argument('--speaker_id', type=int, default=0, help='Speaker ID, e.g., 0.')
    parser.add_argument('--transposition', type=int, default=0, help='Transposition, e.g., 0.')
    parser.add_argument('--f0_method', type=str, default='harvest',
                        help="F0 method, e.g., 'harvest' (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny).")
    parser.add_argument('--crepe_hop_length', type=int, default=160, help='Crepe hop length, e.g., 160.')
    parser.add_argument('--harvest_median_filter_radius', type=int, default=3,
                        help='Harvest median filter radius (0-7), e.g., 3.')
    parser.add_argument('--post_resample_rate', type=int, default=0, help='Post resample rate, e.g., 0.')
    parser.add_argument('--mix_volume_envelope', type=int, default=1, help='Mix volume envelope, e.g., 1.')
    parser.add_argument('--feature_index_ratio', type=float, default=0.78,
                        help='Feature index ratio (0-1), e.g., 0.78.')
    parser.add_argument('--voiceless_consonant_protection', type=float, default=0.33,
                        help='Voiceless Consonant Protection (Less Artifact). Smaller number = more protection. 0.50 means Do not Use. E.g., 0.33.')

    args = parser.parse_args()
    return args


hubert_model = None


def load_hubert(weights_path: str):
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [f"{weights_path}/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


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
        weights_path,
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
            load_hubert(weights_path)
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
    if os.path.exists(f"separated/htdemucs/{track_name}/vocals.wav") and os.path.exists(
            f"separated/htdemucs/{track_name}/no_vocals.wav"):
        print("Found pre-separated track, skipping separation.")
        return track_name
    # Separate the track with demucs
    # Written to /seperated/htdemucs/{track_name}/vocals.wav
    demucs.separate.main(["--two-stems", "vocals", source_audio_path])
    return track_name


def join_track(track_name, model_name):
    # Load the audio files
    vocal = AudioSegment.from_wav(f"{current_dir}/audio-outputs/{track_name}_{model_name}_vocals.wav")
    instrumental = AudioSegment.from_wav(f"separated/htdemucs/{track_name}/no_vocals.wav")

    # Combine the audio files
    combined = vocal.overlay(instrumental)

    return combined


import os
from scipy.io import wavfile


class InferenceManager:
    def __init__(
        self,
        model_name,
        models_path,
        weights_path,
        source_audio_path,
        output_directory='python/inference/RVCv2/audio-outputs',
        feature_index_filepath='logs/mi-test/added_IVF3042_Flat_nprobe_1.index',
        speaker_id=0,
        transposition=0,
        f0_method='harvest',
        crepe_hop_length=160,
        harvest_median_filter_radius=3,
        post_resample_rate=0,
        mix_volume_envelope=1,
        feature_index_ratio=0.78,
        voiceless_consonant_protection=0.33,
    ):
        self.model_name = model_name
        self.models_path = models_path
        self.weights_path = weights_path
        self.source_audio_path = source_audio_path
        self.output_directory = output_directory
        self.feature_index_filepath = feature_index_filepath
        self.speaker_id = speaker_id
        self.transposition = transposition
        self.f0_method = f0_method
        self.crepe_hop_length = crepe_hop_length
        self.harvest_median_filter_radius = harvest_median_filter_radius
        self.post_resample_rate = post_resample_rate
        self.mix_volume_envelope = mix_volume_envelope
        self.feature_index_ratio = feature_index_ratio
        self.voiceless_consonant_protection = voiceless_consonant_protection

        self.status = 'Beginning inference...'
        self.output_filepath = None
        self.finished = threading.Event()

    def find_model(self):
        print("Searching for model...")
        self.pth_file, self.index_file = find_pth_and_index_files(f"{self.models_path}/{self.model_name}/")
        if self.pth_file is None:
            print("No model file found.")
            return
        print("Found model file: %s" % self.pth_file)
        print("Found index file: %s" % self.index_file)
        print("---------------------------------")

    def separate_track(self):
        print("Demucs: Starting track separation...")
        self.track_name = separate_track(self.source_audio_path)
        print("Demucs: Track separation complete.")
        print("---------------------------------")

    def perform_inference(self):
        print("RVCv2: Starting the inference...")
        self.vc_data = get_vc(self.pth_file, {}, {})
        print(self.vc_data)
        print("RVCv2: Performing inference...")
        self.conversion_data = vc_single(
            self.speaker_id,
            f"separated/htdemucs/{self.track_name}/vocals.wav",
            self.transposition,
            None,
            self.f0_method,
            self.index_file if self.index_file is not None else "",
            self.index_file if self.index_file is not None else "",
            self.feature_index_ratio,
            self.harvest_median_filter_radius,
            self.post_resample_rate,
            self.mix_volume_envelope,
            self.voiceless_consonant_protection,
            self.crepe_hop_length,
            self.weights_path,
        )

    def write_files(self):
        if "Success." in self.conversion_data[0]:
            print(
                f"RVCv2: Inference succeeded. Writing to {f'{current_dir}/audio-outputs'}/{f'{self.track_name}_{self.model_name}_vocals.wav'}...")
            wavfile.write(
                f'{f"{current_dir}/audio-outputs"}/{f"{self.track_name}_{self.model_name}_vocals.wav"}',
                self.conversion_data[1][0], self.conversion_data[1][1])
            print(
                f"RVCv2: Finished! Saved output to {f'{current_dir}/audio-outputs'}/{f'{self.track_name}_{self.model_name}_vocals.wav'}")
            print("---------------------------------")
            print("Rejoing the track...")
            self.joined_track = join_track(self.track_name, self.model_name)
            print("Track rejoined.")
            print("Writing completed file...")
            self.joined_track.export(f"{self.output_directory}/{self.track_name}_{self.model_name}.wav", format='wav')
            print("Track successfully written to: " + f"{self.output_directory}/{self.track_name}_{self.model_name}.wav")
            self.output_filepath = f"{self.output_directory}/{self.track_name}_{self.model_name}.wav"
            print("Cleaning up vocal track...")
            os.remove(f"{current_dir}/audio-outputs/{self.track_name}_{self.model_name}_vocals.wav")
            print("---------------------------------")
            print("Inference complete.")
        else:
            print("RVCv2: Inference failed. Here's the traceback: ")
            print(self.conversion_data[0])
    

    def check_and_download_youtube_audio(self, url):
        # Check if the url is a valid YouTube video link
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

        youtube_regex_match = re.match(youtube_regex, url)
        if not youtube_regex_match:
            return False, None

        # Get video id from url
        video_id = youtube_regex_match.group(6)

        # Check if file already exists
        output_path = f'python/inference/RVCv2/audio-outputs/{video_id}.mp3'
        if os.path.exists(output_path):
            return True, output_path

        # Define the options for youtube_dl
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'python/inference/RVCv2/audio-outputs/{video_id}',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        # Use youtube_dl to download the youtube video as an mp3
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            self.status = 'Downloading audio from YouTube...'
            ydl.download([url])
            return True, output_path


    def check_status(self):
        return (self.status, self.output_filepath)
    

    def infer(self):
        self.status = 'Parsing model arguments...'
        self.find_model()
        is_yt_video, yt_audio_path = self.check_and_download_youtube_audio(self.source_audio_path)
        if is_yt_video:
            self.source_audio_path = yt_audio_path
        self.status = 'Separating track...'
        self.separate_track()
        self.status = "Performing inference..."
        self.perform_inference()
        self.status = "Creating audio files..."
        self.write_files()
        self.status = "Complete"
        self.finished.set()

    def run(self):
        threading.Thread(target=self.infer).start()


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
        print("RVCv2: Inference succeeded. Writing to %s/%s..." % (
            f"{current_dir}/audio-outputs", f"{track_name}_{args.model_name}_vocals.wav"))
        wavfile.write(f'{f"{current_dir}/audio-outputs"}/{f"{track_name}_{args.model_name}_vocals.wav"}', conversion_data[1][0], conversion_data[1][1])
        print("RVCv2: Finished! Saved output to %s/%s" % (
            f"{current_dir}/audio-outputs", f"{track_name}_{args.model_name}_vocals.wav"))
        print("---------------------------------")
        print("Rejoing the track...")
        joined_track = join_track(track_name, args.model_name)
        print("Track rejoined.")
        print("Writing completed file...")
        joined_track.export(f"{current_dir}/audio-outputs/{track_name}_{args.model_name}.wav", format='wav')
        print("Track successfully written to: " + f"{current_dir}/audio-outputs/{track_name}_{args.model_name}.wav")
        print("Cleaning up vocal track...")
        os.remove(f"{current_dir}/audio-outputs/{track_name}_{args.model_name}_vocals.wav")
        print("---------------------------------")
        print("Inference complete.")
    else:
        print("RVCv2: Inference failed. Here's the traceback: ")
        print(conversion_data[0])


if __name__ == '__main__':
    main()
