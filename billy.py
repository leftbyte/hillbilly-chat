#!/usr/bin/env python

from datetime import datetime
import openai
import os
import gradio as gr
import typer

"""
Example of a one-turn conversation with a hillbilly chatbot.
"""

VERSION = "1.0.0"


def billy(region: str, input: str):
    """Hillbily chatbot that tranlates phrases to funny old sayings."""
    print(f"date: {datetime.now()}, region: {region}, input: {input}")

    messages = [
        {
            "role": "system",
            "content": f"You are a quirky hillbilly from rural {region} that speaks in colloquialisms.",
        },
    ]

    CONTEXT = "What is a hillbillly colloquialism for: "
    if input:
        messages.append({"role": "user", "content": CONTEXT + input})
        print(f"date: {datetime.now()}, messages: {messages}")
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"date: {datetime.now()}, reply: {reply}")
        return reply


def main():
    apikey = os.getenv("OPENAI_API_KEY")
    openai.api_key = apikey

    inputs = [
        gr.Dropdown(
            [
                "American",
                "Australia",
                "Canada",
                "England",
                "Ireland",
            ],
            label="Colloquial Region",
        ),
        gr.Textbox(lines=5, label="Enter in a phrase to translate:"),
    ]
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
