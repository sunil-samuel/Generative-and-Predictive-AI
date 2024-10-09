from transformers import pipeline
from Transformer.pipeline_base import PipelineBase


class TextClassification(PipelineBase):
    """
    # Text Classification

    Text classification is a common NLP task that assigns a label or class to text.
    There are many practical applications of text classification widely used in
    production by some of today’s largest companies. The `granite-guardian-hap-38m`
    is IBM's lightweight, 4-layer toxicity binary classifier for English. Its
    latency characteristics make it a suitable guardrail for any large language model.
    It can also be used for bulk processing of data where high throughput is needed.
    It has been trained on several benchmark datasets in English, specifically for
    detecting hateful, abusive, profane and other toxic content in plain text.
    
    * **platform** : HuggingFace
    * **url** : https://huggingface.co/tasks/text-classification

    """

    task = "text-classification"
    __model_names = {
        "granite-guardian-hap-38m": "ibm-granite/granite-guardian-hap-38m"
    }

    __pipelines = {}
    __model_name_keys = []

    def __init__(self, device=1) -> None:
        for key, value in self.__model_names.items():
            print(f"{__name__} - Adding models [{key}] => [{value}]")
            self.__pipelines[key] = pipeline(model=value, device=device)
            self.__model_name_keys.append(key)

    def get_model_names(self) -> list[str]:
        return self.__model_name_keys

    def process(self, prompt: str, model_name: str, context: str) -> str:
        print(f"{__name__} - prompt [{prompt}] model name [{model_name}]")

        result = self.__pipelines[model_name](inputs=prompt)
        print(f"{__name__} - result [{result}]")

        return f"[{result[0]["label"]}] = [{result[0]["score"]}]"
