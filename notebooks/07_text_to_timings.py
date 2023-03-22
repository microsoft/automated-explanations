import torchaudio
from tqdm import tqdm
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import joblib
from os.path import join
import imodelsx.util
from mprompt.config import RESULTS_DIR
import json
import stable_whisper
import pandas as pd
import numpy as np

def text_to_speech(text, speech_fname):
    # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="~/.tmpdir_tts")
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="~/.tmpdir_vocoder")

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
    model = stable_whisper.load_model('base')
    result = model.transcribe(speech_fname)
    result.save_as_json(timings_fname_prefix + '.json')
    result.to_srt_vtt(timings_fname_prefix + '.srt')
    # print(result)
    return json.load(open(timings_fname_prefix + '.json', 'r'))

if __name__ == '__main__':
    for EXPT_NAME in ['huth2016clusters_mar21_i_time_traveled', 'voxels_mar21_hands_arms_emergency']:
        EXPT_DIR = join(RESULTS_DIR, 'stories', EXPT_NAME)
        text = open(join(EXPT_DIR, 'story.txt'), 'r').read()

        
        # initialize TTS (tacotron2) and Vocoder (HiFIGAN)
        device = 'cuda'
        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=".tmp/.tmpdir_tts")
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=".tmp/.tmpdir_vocoder")

        # initialize STT
        model = stable_whisper.load_model('base')

        # tmp, only saves 5-sec chunks
        speech_fname = join(EXPT_DIR, 'speech.wav')
        timings_fname_prefix = join(EXPT_DIR, f'{EXPT_NAME}_timings')


        # pass 5 words at a time, each time getting mean time from word 1 to word 0
        # note: could spot-check this by averaging over multiple ngrams
        ngrams_list = imodelsx.util.generate_ngrams_list(text, ngrams=5, pad_starting_ngrams=True)
        timing_running = []
        timing_running2 = []
        for i in tqdm(range(len(ngrams_list))):
            ngram = ngrams_list[i]

            # Running the TTS
            mel_output, mel_length, alignment = tacotron2.encode_text(ngram)

            # Running Vocoder (spectrogram-to-waveform)
            waveforms = hifi_gan.decode_batch(mel_output.to(device))

            # Save the waveform
            torchaudio.save(speech_fname, waveforms.squeeze(1), 22050)

            # Transcribe and save
            result = model.transcribe(speech_fname)
            result.save_as_json(timings_fname_prefix + '.json')
            # result.to_srt_vtt(timings_fname_prefix + '.srt')

            # Load and process
            timings = json.load(open(timings_fname_prefix + '.json', 'r'))
            # print('words_orig', ngram.split())
            word_dicts = timings['segments'][0]['words']
            words = [word_dict['word'].strip() for word_dict in word_dicts]
            timings = [np.mean([word_dict['end'], word_dict['start']]) for word_dict in word_dicts]
            # assert len(ngram.split()) == len(words), f'{ngram} {str(words)}'
            # print('words_new', words)



            if len(words) == 1:
                timing_running.append(timings[0])
                timing_running2.append(timings[0])
            elif len(words) == 2:
                timing_running.append(timings[1] - timings[0])
                timing_running2.append(timings[1] - timings[0])
            else:
                timing_running.append(timings[1] - timings[0])
                timing_running2.append(timings[2] - timings[1])

        pd.DataFrame.from_dict({
            'word': ngrams_list,
            'timing': timing_running, # difference between word1 and word2
            'timing2': timing_running2, # difference between word1 and word2
            'time_running': np.cumsum(timing_running)
        }).to_csv(join(EXPT_DIR, 'timings.csv'))
