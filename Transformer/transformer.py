from Transformer.Pipeline import (
    question_answering,
    sentiment_analysis,
    text_generation,
    text_to_speech,
    image_to_text,
    genai_gemini,
    text_classification,
    meta_llama,
)
from Utilities import cuda_info
import configparser
from huggingface_hub import login
import google.generativeai as genai
import time


class Transformer:
    """
    Create a transformer interface that will interact with all
    """

    Enabled = (True,)

    ###########################################################################
    # constructor
    ###########################################################################
    def __init__(self) -> None:

        self.__pre_construct()

        device = cuda_info.get_gpu()

        self.__context = self.__readContext("context/rhoai_doc.txt")

        self.__pipelines = {
            sentiment_analysis.SentimentAnalysis.task: (
                (sentiment_analysis.SentimentAnalysis(device=device)) if False else None
            ),
            text_generation.TextGeneration.task: (
                (text_generation.TextGeneration(device=device)) if True else None
            ),
            question_answering.QuestionAnswering.task: (
                (question_answering.QuestionAnswering(device=device)) if False else None
            ),
            text_to_speech.TextToSpeech.task: (
                (text_to_speech.TextToSpeech(device=device)) if False else None
            ),
            image_to_text.ImageToText.task: (
                (image_to_text.ImageToText(device=device)) if False else None
            ),
            genai_gemini.GenaiGemini.task: (
                (genai_gemini.GenaiGemini(device=device)) if True else None
            ),
            text_classification.TextClassification.task: (
                (text_classification.TextClassification(device=device))
                if False
                else None
            ),
            meta_llama.MetaLlama.task: (
                (meta_llama.MetaLlama(device=device)) if False else None
            ),
        }

    ###########################################################################
    # get_available_pipelines
    ###########################################################################
    def get_available_pipelines(self):
        """
        Return the list of pipelines that are registered with this transformer.
        """
        rval = []
        for key, value in self.__pipelines.items():
            if value != None:
                rval.append(key)
        rval.sort()
        return rval

    ###########################################################################
    # get_doc
    ###########################################################################
    def get_doc(self, type: str) -> str | None:
        """
        Given the pipeline name, return the doc
        """
        try:
            return self.__find_pipeline(type).__doc__
        except ValueError as error:
            return str(error)

    ###########################################################################
    # get_model_names
    ###########################################################################
    def get_model_names(self, type: str) -> list[str]:
        """
        Given the name of the pipeline, return all of the models that
        are supported by this pipeline
        """
        try:
            pipeline = self.__find_pipeline(type)
            return pipeline.get_model_names()
        except ValueError as error:
            return [str(error)]

    ###########################################################################
    # process
    ###########################################################################
    def process(self, type: str, model_name: str, history: list, prompt: str):
        print(
            f"History is [{history}] type is [{type}] model name [{model_name}] prompt [{prompt}]"
        )
        print(
            f"Calling pipeline [{type}] with prompt [{prompt}] model name [{model_name}] type [{type.__class__}]"
        )

        try:
            pipeline = self.__find_pipeline(type)
            response = pipeline.process(
                prompt=prompt, model_name=model_name, context=self.__context
            )
            yield from self.__bot_response(response, history)
        except ValueError as error:
            yield from self.__bot_response(str(error), history)

    ###########################################################################
    # __readContext
    ###########################################################################
    def __readContext(self, fileName: str) -> str:
        with open(fileName, "r") as file:
            return file.read()

    ###########################################################################
    # __pre_construct
    ###########################################################################
    def __pre_construct(self) -> None:
        config: configparser.ConfigParser = configparser.ConfigParser()
        config.read("config.ini")
        # Log onto huggingface
        login(token=config["huggingface"]["Token"])
        # Log onto Google gemini
        genai.configure(api_key=config["google"]["Token"])

    ###########################################################################
    # __find_pipeline
    ###########################################################################
    def __find_pipeline(self, name: None | str):
        if name == None:
            raise ValueError("Please select a type of pipeline")

        pipeline = self.__pipelines[name]
        if pipeline == None:
            raise ValueError("Invalid pipeline selected.  Please select again.")
        return pipeline

    ###########################################################################
    # __bot_response
    ###########################################################################
    def __bot_response(self, response: str, history: list):
        history[-1][1] = ""

        for character in response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history
