from transformers import pipeline
from Transformer.pipeline_base import PipelineBase

class QuestionAnswering(PipelineBase):
    """
    # Question Answering
    
    Question answering tasks return an answer given a question. If you’ve
    ever asked a virtual assistant like Alexa, Siri or Google what the
    weather is, then you’ve used a question answering model before.
    
    * **platform** : HuggingFace
    * **url** : https://huggingface.co/docs/transformers/en/tasks/question_answering
    """
    __model_names = {
        "distilbert-base-cased-distilled-squad" :
            "distilbert/distilbert-base-cased-distilled-squad",
        "roberta-base-squad2" :
            "deepset/roberta-base-squad2"
    }
    task="question-answering"
    __pipelines = {}
    __model_name_keys = []
   
    def __init__(self, device = -1) -> None:
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
        
    def process (self, prompt:str, model_name:str, context:str) -> str:
        answer = self.__pipelines[model_name] (
            question = prompt,
            context = context
        )
        print (f"{__name__} - prompt [{prompt}] :: response [{answer}]")
        return f"{answer['answer']} : ({answer['score']})"