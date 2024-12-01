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
import random
from sklearn.linear_model import RidgeCV


def text_to_speech(text, speech_fname):
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    tacotron2 = Tacotron2.from_hparams(
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir="~/.tmpdir_tts",
        run_opts={"device": "cuda"},
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir="~/.tmpdir_vocoder",
        run_opts={"device": "cuda"},
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
        source="speechbrain/tts-tacotron2-ljspeech",
        savedir=".tmp/.tmpdir_tts",
        run_opts={"device": "cuda"},
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-ljspeech",
        savedir=".tmp/.tmpdir_vocoder",
        run_opts={"device": "cuda"},
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
        torchaudio.save(speech_fname, waveforms.squeeze(1).cpu(), 22050)

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

    n = len(timing)
    pd.DataFrame.from_dict(
        {
            "word": text.split()[:n],
            # difference between word2 middle and word1 middle
            "timing": timing[:n],
            # 'timing2': timing2[:n], # difference between word3 middle and word2 middle (no offsetting needed)
            "time_running": np.nancumsum(timing),
        }
    ).to_csv(join(EXPT_DIR, "timings.csv"), index=False)


def process_timings(df: pd.DataFrame, min_timing=0.16, target_mean=0.32) -> pd.DataFrame:
    df["word_len"] = df["word"].apply(len)
    df["ends_in_period"] = df["word"].str.endswith(".")
    df["ends_in_comma"] = df["word"].str.endswith(",")

    # truncate values that are too large
    df["timing"] = df["timing"].apply(lambda x: min(x, 0.75))

    # fill na values with linreg
    X = df[["word_len", "ends_in_period", "ends_in_comma"]].values
    y = df["timing"].values
    print("n", y.size, "n_nan", np.sum(pd.isna(y)))

    idxs = ~pd.isna(y)
    m = RidgeCV()
    m.fit(X[idxs], y[idxs])

    # fill na values
    if np.any(idxs):
        df["timing"][~idxs] = m.predict(X[~idxs])

    # fix values that are too small
    idxs = y <= min_timing
    if np.any(idxs):
        df["timing"][idxs] = m.predict(X[idxs])

    # truncate values that are too large
    df["timing"] = np.clip(df["timing"], min_timing, 0.8)

    if df["timing"].mean() < target_mean:
        df["timing"] = df["timing"] * target_mean / df["timing"].mean()

    # remove repeated consecutive words
    df = df[df['word'] != df['word'].shift(1)]

    # recompute running time
    df["time_running"] = np.cumsum(df["timing"])
    return df


if __name__ == "__main__":
    # "polysemantic", "interactions"]:
    # filter = 'may9'
    filter = 'nov30'
    for setting in ['roi']:
        # for setting in ["interactions", "default"]:
        for subject in ["UTS02"]:  # "UTS03"]:
            EXPT_NAMES = sorted(
                [
                    k
                    for k in os.listdir(join(RESULTS_DIR, "stories", setting))
                    if subject.lower() in k.lower()
                    and filter in k
                ]
            )
            # shuffle EXPT_NAMES
            random.shuffle(EXPT_NAMES)
            # EXPT_NAMES = EXPT_NAMES[::-1]

            for EXPT_NAME in tqdm(EXPT_NAMES):
                EXPT_DIR = join(RESULTS_DIR, "stories", setting, EXPT_NAME)

                # get text
                try:
                    prompts_paragraphs = joblib.load(
                        join(EXPT_DIR, "prompts_paragraphs.pkl"),
                    )
                    text = "\n".join(prompts_paragraphs["paragraphs"])
                except:
                    rows = joblib.load(join(EXPT_DIR, "rows.pkl"))
                    text = "\n".join(rows.paragraph.values)

                timings_file = join(EXPT_DIR, "timings.csv")

                # get timings for timings_file
                if os.path.exists(timings_file):
                    print("already cached", EXPT_NAME)
                else:
                    print("running", EXPT_NAME)
                    process_story(
                        EXPT_DIR,
                        EXPT_NAME,
                        text,
                        timings_fname_prefix=join(
                            EXPT_DIR, f"{EXPT_NAME}_timings"),
                    )

                # process timings
                df = pd.read_csv(timings_file)
                df = process_timings(df)
                df.to_csv(join(EXPT_DIR, "timings_processed.csv"), index=False)
