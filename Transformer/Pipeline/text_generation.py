from transformers import pipeline
from Transformer.pipeline_base import PipelineBase

class TextGeneration(PipelineBase):
    """
    # Text Generation

    Generating text is the task of generating new text given another text.
    These models can, for example, fill in incomplete text or paraphrase.

    * **platform** : HuggingFace
    * **url** : https://huggingface.co/tasks/text-generation
    """

    task = "text-generation"
    __model_names = {
        "openai-gpt2": "openai-community/gpt2",
        "gemma-2-2b-it": "google/gemma-2-2b-it",
    }

    __model_kwargs = {
        "openai-gpt2": {"max_length": 30, "num_return_sequences": 5},
        "gemma-2-2b-it": {"max_new_tokens": 1024},
    }
    __pipelines = {}
    __model_name_keys = []

    def __init__(self, device=-1) -> None:
        __model_names = []
        for key, value in self.__model_names.items():
            print(
                f"{__name__} - Adding models [{key}] => [{value}] [{self.__model_kwargs[key]}]"
            )
            self.__pipelines[key] = pipeline(
                task=self.task,
                model=value,
                device=device,
                do_sample=True,
            )
            self.__model_name_keys.append(key)

    def get_model_names(self) -> list[str]:
        return self.__model_name_keys

    def process(self, prompt: str, model_name: str, context: str) -> str:
        my_args: dict = {"text_inputs": prompt}
        my_args.update(self.__model_kwargs[model_name])

        kwargs = self.__pipelines[model_name](**my_args)
        print(f"{__name__} - prompt [{prompt}] :: response [{kwargs}]")

        rval: str = ""

        for index in range(0, len(kwargs)):
            rval += f"## Response {index+1}\n\n"
            rval += f"* {kwargs[index]['generated_text']}\n\n"
        return rval
