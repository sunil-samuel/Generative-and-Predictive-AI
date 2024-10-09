<p align='right'>
	<small>Sunil Samuel<br>
		web_github@sunilsamuel.com<br>
		http://www.sunilsamuel.com
	</small>
</p>

# Predicative and GenAI

The objective of this project was to understand different concepts of machine learning models, including Large Language Models (LLMs), Multimodal models, generative models, and others.  It compares multiple types of models and pipelines and uses different different APIs, such as Huggingface and Google GenAI.

## Prerequisite

The code is written in **Python**, therefore understanding Python is a requirement.
><img src="docs/gfx/note.png" style="width:20px;display:inline-block;vertical-align:middle;"> Visual Studio Code is the IDE used as the development environment.  If you are developing on Windows environment, then install the **Remote Development** extension and develop within the WSL Ubuntu environment.

### Python Virtual Environment

Although not required, creating a virtual environment will help with decoupling this environment with other Python environments.  Install the virtual environment as follows:

1. Select a directory where the virtual environment is created and run the following command.
    > `python -m venv /path/to/new/virtual/environment`
1. Change to the new environment.
    > `source /path/to/new/virtual/environment/bin/activate.csh`
1. Install the modules within the `requirements.txt` file.
    > `pip install -r requirements.txt`

### Generate Tokens

In order to use the models from Hugging Face and Google, you must generate tokens for each.

1. Google - https://aistudio.google.com/app/apikey
1. Hugging Face - https://huggingface.co/settings/tokens

### Update the `config.ini` file

Use the tokens created to update the `config.ini` file in the root of this project.

```ini
# https://huggingface.co/settings/tokens
[huggingface]
Token = hf_1lasdj3l23lkjsfasdkDlsdBdksdfuvlka

# https://aistudio.google.com/app/apikey
[google]
Token = AIaslkdfadC723234kDfklFkwrewLkUsfjfg12j
```

## Technology Stack

The following technology is used to create the project.

1. **HuggingFace Pipelines**
    > Wrapper that makes it easy to use models for inference.  Pipelines are objects that abstract most of the complex code.
    >
    > https://huggingface.co/docs/transformers/en/main_classes/pipelines
1. **HuggingFace Transformers**
    > APIs and tools that faciliates downloading and training pretrained models.
    >
    > https://huggingface.co/docs/transformers/en/index
1. **Google Generative AI**
    > Python SDK to access the Gemini API that allows access to the Gemini models.
    >
    > https://github.com/google-gemini/generative-ai-python
1. **psutil**
    > Retrieve information on running processes and system utilization, such as CPU, memory, ...
    >
    > https://pypi.org/project/psutil/
1. **torch**
    > Contains data structures and operations over tensors.
    >
    > https://pypi.org/project/torch/
1. **gradio**
    > Package that helps to quickly build a demo or a web application for ML.
    >
    > https://www.gradio.app/

## Code

The code is written in and tested using Python (3.12.3) and uses the Model-View-Controller design pattern.  The Model is responsibible for the backend processing, View is responsible for the frontend interface, and the Controller faciliates the communication between the two (model and view).

1. **model** - `pipeline_base.py` (backend invocation of LLMs)
1. **view** - `ui_interface.py` (web interface)
1. **controller** - `transformer.py` (uses the front end data and calls the backend LLMs)

><img src="docs/gfx/note.png" style="width:20px;display:inline-block;vertical-align:middle;"> The number of LLMs you can load is based on the size and resources of your computer.  If you run into memory issues, then disable some of the LLMs within the constructor of the `transformer.py` code.
>
> To disable an LLM, open `transformer.py` and set pipelines to `False`.
>

```python
self.__pipelines = {
    # This is disabled given the `if False`
    sentiment_analysis.SentimentAnalysis.task: (
        sentiment_analysis.SentimentAnalysis(device=device) 
    ) if False else None,
    # This is enabled given the `if True`
    text_generation.TextGeneration.task: (
        text_generation.TextGeneration(device=device) 
    ) if True else None,
    ...
}
```

### Adding Models

Additional models can be added by :

1. implementing the abstract class `PipelineBase`
1. creating docstring for the class
1. use the new models, see existing pipeline codes for `Pipeline` package

### Running Application

To run the application:

```sh
sh>/<my virtual environment>/bin/python "/<dir to code>/ui_interface.py"
```

Open a browser to the following URL:

http://localhost:7860/