import numpy as np
import soundfile as sf
import pydub

def read_soundfile(file):
    samples, origin_sr = sf.read(file)
    samples = samples[:, np.newaxis].astype(np.float32)
    return samples, origin_sr

def read_pydub(file):
    try:
        asegment = pydub.AudioSegment.from_file(file)
        origin_sr = asegment.frame_rate
        channel_sounds = asegment.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]
        # Convert to float32
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        return fp_arr, origin_sr
    except pydub.exceptions.CouldntDecodeError:
        return None, None