from transformers import pipeline
from Transformer.pipeline_base import PipelineBase

class SentimentAnalysis(PipelineBase):
    """
    # Sentiment Analysis
    
    Sentiment analysis uses Natural Language Processing (NLP) to determine the tone
    of the message to be positive, negative, or neutral.
    
    * **platform** : HuggingFace
    * **url** : https://huggingface.co/blog/sentiment-analysis-python
    """
    __model_names = {
        "distilbert-base-uncased-finetuned-sst-2-english" : 
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "bertweet-base-sentiment-analysis" :
            "finiteautomata/bertweet-base-sentiment-analysis"
    }
    
    __pipelines = {}
    __model_name_keys = []
    task="sentiment-analysis"
    
    def __init__(self, device=-1) -> None:
        for key, value in self.__model_names.items():
            print(f"{__name__} - Adding models [{key}] => [{value}]")
            self.__pipelines[key] = pipeline (
                task=self.task,
                model=value,
                device=device
            )
            self.__model_name_keys.append(key)
                
    def get_model_names (self) -> list[str]:
        return self.__model_name_keys
        
    def process(self, prompt:str, model_name:str, context:str) -> str:
        print (f"{__name__} - prompt [{prompt}] - model name [{model_name}]")
        classifier= self.__pipelines[model_name](prompt)
        print (f"{__name__} - prompt [{prompt}] - model name [{model_name}] :: response [{classifier}]")
        return f"{classifier[0]['label']} : {classifier[0]['score']}"
