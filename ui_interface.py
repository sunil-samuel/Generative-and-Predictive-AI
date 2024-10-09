import gradio
from Transformer.transformer import Transformer
from Utilities import cuda_info


def create_ui():
    """Create the UI using gradio"""
    with gradio.Blocks(
        theme="glass",
        title="Interface to Invoke Pipelines",
        css=get_css(),
        fill_height=True,
        fill_width=True,
    ) as web_interface:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  L A Y O U T - SECTION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #
        # The description at the top of the page
        #
        gradio.Markdown(value=get_description())
        #
        # The chatbot that prints the user and system outputs
        #
        chatbot = gradio.Chatbot(placeholder="Output", avatar_images=None)

        with gradio.Row():
            #
            # Dropdown selection for the pipeline
            #
            task_selection: gradio.Dropdown = gradio.Dropdown(
                label="Select the type of pipeline (model)",
                allow_custom_value=False,
                choices=transformer.get_available_pipelines(),
            )

            model_selection: gradio.Dropdown = gradio.Dropdown(
                value="Select a 'type of pipeline' first",
                label="Select a model after selecting a pipeline",
                allow_custom_value=True,
                choices=["Select a 'type of pipeline' first"],
                interactive=True,
            )
        #
        # Accordion that provides additional information on the
        # pipeline chosen by the user.
        #
        with gradio.Accordion(label="See Details"):
            task_selection_description = gradio.Markdown("")
        #
        # Textbox where the user will enter the prompt to be sent
        # to the model (pipeline) to be invoked.
        #
        user_input: gradio.Textbox = gradio.Textbox(
            autofocus=True,
            label="Enter the prompt:",
            placeholder="How long can I use RHOAI on a trial basis?",
        )
        #
        # Button that will clear (reset) the form completely
        #
        clear = gradio.Button("Clear Everything")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  E V E N T - H A N D L E R - SECTION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        def prompt_update(selection, user_message, history):
            return f"{user_message}", history + [[f"{selection}: {user_message}", None]]

        user_input.submit(
            fn=prompt_update,
            inputs=[task_selection, user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(
            fn=transformer.process,
            inputs=[task_selection, model_selection, chatbot, user_input],
            outputs=chatbot,
        )

        clear.click(
            lambda: (None, None, None),
            inputs=None,
            outputs=[user_input, chatbot, task_selection],
            queue=False,
        )

        def task_selection_change(type):
            print(f"{__name__} - task_selection_change :: Type [{type}]")
            return "" if (type == None) else transformer.get_doc(type)

        task_selection.change(
            task_selection_change,
            inputs=task_selection,
            outputs=task_selection_description,
        )

        def task_selection_model_change(type: str | None) -> list | gradio.Dropdown:
            """Based on the user selection, change to the correct model.

            Args:
                type (str | None): name of the model

            Returns:
                list | gradio.Dropdown: The dropdown with the correct model selection
            """
            model_names: list[str] = []
            if type == None:
                model_names.append("Select a 'type of pipeline' first"),  # type: ignore
                value = "Select a 'type of pipeline' first"
            else:
                model_names = transformer.get_model_names(type)
                value = model_names[0]
            print(
                f"{__name__} - Type [{type}] model names [{model_names}] :: [{type.__class__}]"
            )
            return gradio.Dropdown(choices=model_names, interactive=True, value=value)

        task_selection.change(
            task_selection_model_change, inputs=task_selection, outputs=model_selection
        )

    web_interface.launch()


def get_description() -> str:
    return (
        """
# Objectives

Huggingface uses **pipelines** as opposed to **langchain**.

> ## What is the difference between LangChain and Hugging Face pipeline?

> LangChain is a framework for building NLP pipelines. It offers tools for
> data processing, model integration (including Hugging Face models), and 
> workflow management. Hugging Face pipelines are pre-built wrappers for 
> specific NLP tasks that can be used within LangChain or other environments.

## What is a huggingface pipeline

The pipelines are a great and easy way to use models for inference. These
pipelines are objects that abstract most of the complex code from the
library, offering a simple API dedicated to several tasks, including
Named Entity Recognition, Masked Language Modeling, Sentiment Analysis,
Feature Extraction and Question Answering.

> * [HuggingFace Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines)

## Relevant Links
    
> * [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline)
> * [Google Models](https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models)
> * [HuggingFace Model List](https://huggingface.co/models)
> * [Granite Model on Huggingface List](https://huggingface.co/ibm-granite)
"""
        + cuda_info.print_gpu()
    )


def get_css() -> str:
    return """

* {
    font-family: sans-serif !important;
}

footer {
    display: none !important;
}

.md {
    opacity: 1;
}
"""


print(f"{cuda_info.print_gpu()}")
transformer = Transformer()
create_ui()
