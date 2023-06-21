#!/usr/bin/env python

import openai
import os
import gradio as gr
import typer

"""
Example of a one-turn conversation with a hillbilly chatbot.
"""

VERSION = "1.0.0"


def billy(input: str = None):
    """Hillbily chatbot that tranlates phrases to funny old sayings."""
    messages = [
        {
            "role": "system",
            "content": "You are a quirky hillbilly from the 1920s that speaks in colloquialisms.",
        },
    ]

    CONTEXT = "What is a hillbillly colloquialism for: "
    if input:
        messages.append({"role": "user", "content": CONTEXT + input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply


def main():
    apikey = os.getenv("OPENAI_API_KEY")
    openai.api_key = apikey

    inputs = gr.Textbox(lines=7, label="Enter in a phrase to translate:")
    outputs = gr.Textbox(label="I reckon I'd say:")

    gr.Interface(
        fn=billy,
        inputs=inputs,
        outputs=outputs,
        title=f"Colloquial Chatterbot {VERSION}",
        description="Things a hillbilly would say",
        theme=gr.themes.Monochrome(),
    ).launch(share=True)


if __name__ == "__main__":
    typer.run(main)
