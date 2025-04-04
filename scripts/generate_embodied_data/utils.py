import json
import os
import time
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    

class Gemini:
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.environ.get("GEMINI_API_KEY", None)
        assert api_key is not None, "GEMINI_API_KEY is not set"
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.init_model(model_name)

    def init_model(self, model_name):
        self.model = genai.GenerativeModel(model_name)

    def safe_call(self, f):
        while True:
            try:
                res = f()
                return res
            except ResourceExhausted:
                time.sleep(5)

    def generate(self, prompt):
        chat = self.safe_call(lambda: self.model.start_chat(history=[]))
        response = self.safe_call(lambda: chat.send_message(prompt).text)

        for i in range(8):
            if response is None:
                print(f"n_retries: {i}")
                return None
            if "FINISHED" in response:
                print(f"n_retries: {i}")
                return response
            else:
                print("FINISHED not found in response")
            response = response + self.safe_call(lambda: chat.send_message("Truncated, please continue.").text)

        print(f"n_retries: {i}")

        return None


def post_process_caption(caption, lang_instruction):
    text = caption.replace(",", ".")
    if text[-1] != ".":
        text += "."
    return text
