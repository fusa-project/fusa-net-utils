import numpy as np
import soundfile as sf
import pydub
import logging

logger = logging.getLogger(__name__)

def read_soundfile(file):
    try:
        samples, origin_sr = sf.read(file)
        samples = samples[:, np.newaxis].astype(np.float32)
        logger.info("soundfile read sucessfully")
        return samples, origin_sr
    except Exception as error:
        logger.info(f"Error en soundfile: {error}")
        return None, None

def read_pydub(file):
    try:
        asegment = pydub.AudioSegment.from_file(file)
        logger.info("pydub read sucessfully")
        origin_sr = asegment.frame_rate
        channel_sounds = asegment.split_to_mono()
        #logger.info(f"channel_sounds len: {len(channel_sounds)}")
        samples = [s.get_array_of_samples() for s in channel_sounds]
        #logger.info("first array_of_samples:", len(channel_sounds[0].get_array_of_samples()))
        #logger.info("len of samples :",len(samples))
        # Convert to float32
        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        return fp_arr, origin_sr
    except pydub.exceptions.CouldntDecodeError:
        return None, None
    except Exception as error:
        logger.info(f"Error en pydub: {error}")
