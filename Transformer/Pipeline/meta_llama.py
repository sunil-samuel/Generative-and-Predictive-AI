from typing import Any
from Transformer.pipeline_base import PipelineBase
from transformers import pipeline
import torch

class MetaLlama(PipelineBase):
    """
    The Meta Llama 3.1 collection of multilingual large language 
    models (LLMs) is a collection of pretrained and instruction 
    tuned generative models.
    
     * **model** : Meta-Llama-3.1-8B-Instruct
    * **platform** : Meta
    * **url** : https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   
    """
    __model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    task = "meta-llama-instruct"

    def __init__(self, device=-1):
        self.__pipeline = pipeline(
            model=self.__model_name,
            model_kwargs={"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True},
            device=device,
        )

    def get_model_names(self) -> list[str]:
        return [self.__model_name]

    def process(self, prompt: str, model_name: str, context: str) -> str:
        response:Any = self.__pipeline(
            prompt, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=1
        )
        return response[0]['generated_text'][-1]