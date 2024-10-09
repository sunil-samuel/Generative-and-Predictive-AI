from transformers import pipeline
from Transformer.pipeline_base import PipelineBase


class ImageToText(PipelineBase):
    """
    # Image to Text

    Image to text models output a text from a given image. Image captioning
    or optical character recognition can be considered as the most common
    applications of image to text.

    * **platform** : HuggingFace
    * **url** : https://huggingface.co/tasks/image-to-text
    """

    __model_names = {
        "vit-gpt2-coco-en": "ydshieh/vit-gpt2-coco-en",
        "blip-image-captioning-base": "Salesforce/blip-image-captioning-base",
    }

    __pipelines = {}
    __model_name_keys = []
    task = "image-to-text"

    def __init__(self, device=-1) -> None:
        for key, value in self.__model_names.items():
            print(f"{__name__} - Adding models [{key}] => [{value}]")
            self.__pipelines[key] = pipeline(task=self.task, model=value, device=device)
            self.__model_name_keys.append(key)

    def get_model_names(self) -> list[str]:
        return self.__model_name_keys

    def process(self, prompt: str, model_name: str, context: str) -> str:
        print(f"{__name__} - prompt [{prompt}]")

        try:
            captioner = self.__pipelines[model_name](prompt)
            return captioner[0]["generated_text"]
        except ValueError as error:
            print(f"Input is not url {error}")
            return f"Please make sure the URL is valid. {error}."