from transformers import pipeline
import scipy.io.wavfile as write_wav
from Transformer.pipeline_base import PipelineBase

class TextToSpeech(PipelineBase):
    """
    # Text to Speech
    
    Text-to-Speech (TTS) is the task of generating natural sounding speech
    given text input. TTS models can be extended to have a single model
    that generates speech for multiple speakers and multiple languages.

    * **platform** : HuggingFace
    * **url** : https://huggingface.co/tasks/text-to-speech
    """
    __model_name="suno/bark"
    task="text-to-speech"
    
    def __init__(self, device=-1) -> None:
        self.__pipeline = pipeline(
            task=self.task,
            model = self.__model_name,
            device=device
        )
        
    def get_model_names(self) -> list[str]:
        return [self.__model_name]
    
    def process (self, prompt:str, context:str) -> str:
        speech = self.__pipeline(
            prompt,
            forward_params={"do_sample": True}
        )
        print (f"{__name__} - prompt [{prompt}] :: response [{speech}]")
                        
        write_wav.write("bark_small_output.wav", rate=speech["sampling_rate"], data=speech["audio"][0]) # type: ignore

        

        #audio_array = speech.cpu().numpy().squeeze()
        #audio_array /=1.414
        #audio_array *= 32767
        #audio_array = audio_array.astype(np.int16)
# print(audio_array)

        #scipy.io.wavfile.write("bark_out_bet.wav", rate=speech["sampling_rate"], data=audio_array)

        
        #audio_array = speech["audio"].cpu().numpy().squeeze();
        #audio_array = speech["audio"][0]
        #audio_array /=1.414
        #audio_array *= 32767
        #audio_array = audio_array.astype(np.float32)
        
        #data = speech['audio']
        #data2 = []

        #for i in range(len(data)):
        #    data2.append([int(round(math.sin(data[i][0])*3000)), int(round(math.sin(data[i][1])*3000))])


        
        #data2 = np.asarray(data2, dtype=np.int16)

        
        #print (f"{__name__} - prompt [{prompt}] :: response [{speech}]")
        #scipy.io.wavfile.write("speech_out.wav", rate=speech["sampling_rate"], data=audio_array)
        #scipy.io.wavfile.write("speech_out.wav", rate=speech["sampling_rate"], data=  np.hstack(data2))
        #scipy.io.wavfile.write("speech_out.wav", rate=speech["sampling_rate"], data=audio_array)
        #scipy.io.wavfile.write("speech_out.wav", rate=speech["sampling_rate"], data=data2)
        #scipy.io.wavfile.write("speech_out.wav", rate=speech["sampling_rate"], data=np.array(data))
        return "wrote file"