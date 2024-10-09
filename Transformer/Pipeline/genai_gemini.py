import google.generativeai as genai
import google.generativeai.types.safety_types as safety_types

from Transformer.pipeline_base import PipelineBase


class GenaiGemini(PipelineBase):
    """
    # Gemini 1.5 Flash

    Gemini 1.5 Flash is a multimodal model that can be used to summarize, analyze,
    and generate content from different context, such as emails, files, ...

    * **model** : gemini-1.5-flash
    * **platform** : Google Generative AI
    * **url** : https://deepmind.google/technologies/gemini/flash/
    """

    __model_name = "gemini-1.5-flash"
    task = "genai-gemini-1.5-flash"

    def __init__(self, device=-1) -> None:
        self.__generation_config = genai.GenerationConfig(
            # max_output_tokens:
            # The maximum number of tokens to include in a candidate.
            #
            # Specifies the maximum number of tokens that can be generated
            # in the response. A token is approximately four characters.
            # 100 tokens correspond to roughly 60-80 words.
            max_output_tokens=2048,
            # temperature:
            # Controls the randomness of the output. Note: The
            # default value varies by model, see the `Model.temperature`
            # attribute of the `Model` returned the `genai.get_model`
            # function.
            #
            # Values can range from [0.0,1.0], inclusive. A value closer
            # to 1.0 will produce responses that are more varied and
            # creative, while a value closer to 0.0 will typically result
            # in more straightforward responses from the model.
            #
            # The temperature controls the degree of randomness in token
            # selection. The temperature is used for sampling during
            # response generation, which occurs when topP and topK are
            # applied. Lower temperatures are good for prompts that require
            # a more deterministic or less open-ended response, while higher
            # temperatures can lead to more diverse or creative results.
            # A temperature of 0 is deterministic, meaning that the highest
            # probability response is always selected.
            temperature=0.4,
            # top_p:
            # Optional. The maximum cumulative probability of tokens to
            # consider when sampling.
            #
            # The model uses combined Top-k and nucleus sampling.
            #
            # Tokens are sorted based on their assigned probabilities so
            # that only the most likely tokens are considered. Top-k
            # sampling directly limits the maximum number of tokens to
            # consider, while Nucleus sampling limits number of tokens
            # based on the cumulative probability.
            #
            # Note: The default value varies by model, see the
            # `Model.top_p` attribute of the `Model` returned the
            # `genai.get_model` function.
            #
            # topP: The topP parameter changes how the model selects
            # tokens for output. Tokens are selected from the most to
            # least probable until the sum of their probabilities equals
            # the topP value. For example, if tokens A, B, and C have a
            # probability of 0.3, 0.2, and 0.1 and the topP value is 0.5,
            # then the model will select either A or B as the next token by
            # using the temperature and exclude C as a candidate.
            # The default topP value is 0.95.
            top_p=1,
            # top_k (int):
            # Optional. The maximum number of tokens to consider when
            # sampling.
            #
            # The model uses combined Top-k and nucleus sampling.
            #
            # Top-k sampling considers the set of `top_k` most probable
            # tokens. Defaults to 40.
            #
            # Note: The default value varies by model, see the
            # `Model.top_k` attribute of the `Model` returned the
            # `genai.get_model` function.
            #
            # topK: The topK parameter changes how the model selects tokens
            # for output. A topK of 1 means the selected token is the most
            # probable among all the tokens in the model's vocabulary
            # (also called greedy decoding), while a topK of 3 means
            # that the next token is selected from among the 3 most
            # probable using the temperature. For each token selection
            # step, the topK tokens with the highest probabilities
            # are sampled. Tokens are then further filtered based on
            # topP with the final token selected using temperature sampling.
            top_k=32,
        )

        self.__safety_settings = [
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]

        self.__model = genai.GenerativeModel(model_name=self.__model_name)

    def process(self, prompt: str, model_name: str, context: str) -> str:
        response = self.__model.generate_content(
            prompt,
            generation_config=self.__generation_config,
            safety_settings=self.__safety_settings,
            # stream=True
        )

        return response.text

    def get_model_names(self) -> list[str]:
        return [self.__model_name]
