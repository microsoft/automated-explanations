# pip install torchaudio speechbrain stable-ts

import torchaudio
from tqdm import tqdm
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import joblib
from os.path import join
import imodelsx.util
from sasc.config import RESULTS_DIR
import json
import stable_whisper
import pandas as pd
import os.path
import numpy as np
import fire
import random


def text_to_speech(text, speech_fname):
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech", savedir="~/.tmpdir_tts"
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech", savedir="~/.tmpdir_vocoder"
    )

    # Running the TTS
    mel_output, mel_length, alignment = tacotron2.encode_text(text)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_output)

    # Save the waverform
    torchaudio.save(speech_fname, waveforms.squeeze(1), 22050)
    # print('done!')

    # from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    # import torch
    # import soundfile as sf
    # from datasets import load_dataset

    # processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    # model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    # vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

    # # load xvector containing speaker's voice characteristics from a dataset
    # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # sf.write("speech.wav", speech.numpy(), samplerate=16000)


def speech_to_text(speech_fname, timings_fname_prefix):
    # import whisper
    # model = whisper.load_model("medium.en")
    # stable_whisper gives better word timings
    model = stable_whisper.load_model("base")
    result = model.transcribe(speech_fname)
    result.save_as_json(timings_fname_prefix + ".json")
    result.to_srt_vtt(timings_fname_prefix + ".srt")
    # print(result)
    return json.load(open(timings_fname_prefix + ".json", "r"))


def process_story(EXPT_DIR, EXPT_NAME, text, timings_fname_prefix):
    # initialize TTS (tacotron2) and Vocoder (HiFIGAN)
    # device = 'cuda'
    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech", savedir=".tmp/.tmpdir_tts"
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech", savedir=".tmp/.tmpdir_vocoder"
    )

    # initialize STT
    model = stable_whisper.load_model("base")

    # tmp, only saves 5-sec chunks
    speech_fname = join(EXPT_DIR, "speech.wav")
    timings_fname_prefix = join(EXPT_DIR, f"{EXPT_NAME}_timings")

    # pass 5 words at a time, each time getting mean time from word 1 to word 0
    # note: could spot-check this by averaging over multiple ngrams
    ngrams_list = imodelsx.util.generate_ngrams_list(
        text, ngrams=3, pad_ending_ngrams=True
    )
    timing = []
    # timing2 = [np.nan] # offset by one word
    for i in tqdm(range(len(ngrams_list))):
        ngram = ngrams_list[i]

        # Running the TTS
        mel_output, mel_length, alignment = tacotron2.encode_text(ngram)

        # Running Vocoder (spectrogram-to-waveform)
        waveforms = hifi_gan.decode_batch(mel_output)

        # Save the waveform
        torchaudio.save(speech_fname, waveforms.squeeze(1), 22050)

        try:
            # Transcribe and save
            result = model.transcribe(speech_fname)
            result.save_as_json(timings_fname_prefix + ".json")
            # result.to_srt_vtt(timings_fname_prefix + '.srt')

            # Load and process
            timings = json.load(open(timings_fname_prefix + ".json", "r"))
            # print('words_orig', ngram.split())

            word_dicts = timings["segments"][0]["words"]
            words = [word_dict["word"].strip() for word_dict in word_dicts]
            timings = [
                np.mean([word_dict["end"], word_dict["start"]])
                for word_dict in word_dicts
            ]
            # assert len(ngram.split()) == len(words), f'{ngram} {str(words)}'
            # print('words_new', words)
            # print(ngram, timings)

            # timing of the first word
            timing.append(timings[1] - timings[0])
            # timing2.append(timings[2] - timings[1])
        except:
            timing.append(np.nan)
            # timing2.append(np.nan)
            # print(ngram, timings)

        # if i % 20 == 0:
        #     # save early
        #     n = len(timing)
        #     pd.DataFrame.from_dict({
        #         'word': text.split()[:n],
        #         'timing': timing[:n], # difference between word2 middle and word1 middle
        #         # 'timing2': timing2[:n], # difference between word3 middle and word2 middle (no offsetting needed)
        #         'time_running': np.nancumsum(timing)
        #     }).to_csv(join(EXPT_DIR, 'timings.csv'), index=False)

    n = len(timing)
    pd.DataFrame.from_dict(
        {
            "word": text.split()[:n],
            "timing": timing[:n],  # difference between word2 middle and word1 middle
            # 'timing2': timing2[:n], # difference between word3 middle and word2 middle (no offsetting needed)
            "time_running": np.nancumsum(timing),
        }
    ).to_csv(join(EXPT_DIR, "timings.csv"), index=False)


if __name__ == "__main__":
    setting = "default"
    # setting = "polysemantic"
    # setting = "interactions"
    EXPT_NAMES = os.listdir(join(RESULTS_DIR, "pilot_v1", setting))
    # shuffle EXPT_NAMES
    random.shuffle(EXPT_NAMES)

    for EXPT_NAME in tqdm(EXPT_NAMES):
        EXPT_DIR = join(RESULTS_DIR, "pilot_v1", setting, EXPT_NAME)
        if setting in ["polysemantic", "default"]:
            rows = joblib.load(join(EXPT_DIR, "rows.pkl"))
            try:
                prompts_paragraphs = joblib.load(
                    join(EXPT_DIR, "prompts_paragraphs.pkl"),
                )
                text = "\n".join(prompts_paragraphs["paragraphs"])
            except:
                text = "\n".join(rows.paragraph.values)
        elif setting == "interactions":
            prompts_paragraphs = joblib.load(
                join(EXPT_DIR, "prompts_paragraphs.pkl"),
            )
            text = "\n".join(prompts_paragraphs["paragraphs"])

        if os.path.exists(join(EXPT_DIR, "timings.csv")):
            print("already cached", EXPT_NAME)
            continue
        else:
            print("running", EXPT_NAME)
            process_story(
                EXPT_DIR,
                EXPT_NAME,
                text,
                timings_fname_prefix=join(EXPT_DIR, f"{EXPT_NAME}_timings"),
            )
