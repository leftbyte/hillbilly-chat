#!/usr/bin/env python

import datetime
import hashlib
import openai
import os
import gradio as gr
import typer
from typing import List
from typing import TypedDict

"""
Example of a one-turn conversation with a hillbilly chatbot.
"""

VERSION = "1.0.0"

# We have these as globals because the inputs to the handler won't accept ints.
ALL_REQ_LIMIT_HITS = 24
CLIENT_REQ_LIMIT_HITS = 6
REQ_LIMIT_MINUTES = 1
CLIENTS = {}
ALL_CLIENTS_TS = []


def limitReached(
    history: List[datetime.datetime],
    limitHits: int,
    limitMinutes: int,
    ts: datetime.datetime,
) -> bool:
    """
    Check whether the limit has been reached. This function mutates
    the input history List
    """
    timeLimit = ts - datetime.timedelta(minutes=limitMinutes)
    for d in history:
        if d < timeLimit:
            history.remove(d)

    # print(
    #     f"history_len: {len(history)}, limit_reached: {len(history) > limitHits}"
    # )
    return len(history) > limitHits


def checkClient(clients: TypedDict, requester: str) -> (bool, str):
    now = datetime.datetime.now()
    hsh = hashlib.sha256(bytes(requester, "UTF-8")).hexdigest()
    if hsh not in clients:
        print(f"requester: {requester}")
        clients[hsh] = [now]
    else:
        clients[hsh].append(now)
        if limitReached(
            clients[hsh], CLIENT_REQ_LIMIT_HITS, REQ_LIMIT_MINUTES, now
        ):
            print(f"date: {now}, client: {hsh}, message: 'limit reached'")
            return True, "ERROR: REQUESTS PER MINIUTE LIMIT REACHED"

    # check the global clients request rate
    ALL_CLIENTS_TS.append(now)
    if limitReached(ALL_CLIENTS_TS, ALL_REQ_LIMIT_HITS, REQ_LIMIT_MINUTES, now):
        return True, "ERROR: ALL API REQUESTS PER MINIUTE LIMIT REACHED"

    return False, ""


def getRequester(request: gr.Request) -> str:
    requester = request.client.host
    if "user-agent" in request.headers:
        requester = "".join([requester, "::", request.headers["user-agent"]])

    return requester


def billy(
    region: str,
    input: str,
    request: gr.Request,
) -> str:
    """Hillbily chatbot that tranlates phrases to funny old sayings."""
    print(f"date: {datetime.datetime.now()}, region: {region}, input: {input}")

    requester = getRequester(request)
    err, msg = checkClient(CLIENTS, requester)
    if err:
        return msg

    messages = [
        {
            "role": "system",
            "content": f"You are a quirky hillbilly from rural {region} that speaks in colloquialisms.",
        },
    ]

    CONTEXT = "What is a hillbillly colloquialism for: "
    if input:
        messages.append({"role": "user", "content": CONTEXT + input})
        print(f"date: {datetime.datetime.now()}, messages: {messages}")
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"date: {datetime.datetime.now()}, reply: {reply}")
        return reply


def main():
    apikey = os.getenv("OPENAI_API_KEY")
    openai.api_key = apikey

    # "Appalachia",
    # "Breadbasket",
    # "Ecotopia",
    # "Dixie",
    # "Foundry",
    # "Mexamerica",
    # "New England",

    inputs = [
        gr.Dropdown(
            [
                "Australia",
                "Canada",
                "England",
                "Ireland",
                "United States",
            ],
            label="Colloquial Region",
            value="United States",
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
    ).launch(share=False)


if __name__ == "__main__":
    typer.run(main)
