# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
# This module provides the timer for LLM inference.
# The Timer class used in this work was originally used for MIRA.

import time
from transformers import TextStreamer

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, ecx_type, exc_value, traceback):
        self.elapsed_time = time.perf_counter() - self.start_time
        return self.elapsed_time

    def __float__(self):
        if self.elapsed_time is None:
            raise ValueError("Timer has not been started or is still running.")
        return self.elapsed_time * 1000   # millisecond

# Define a custom TextStreamer to capture token generation times
class TimingTextStreamer(TextStreamer):
    """
    Custom TextStreamer to capture token generation times
    This class overrides the put method to capture token generation times
    """
    def __init__(self, tokenizer, start_generation_time):
        super().__init__(tokenizer)
        self.start_generation_time = start_generation_time
        self.token_generation_times = []
        self.inter_token_times = []
        self.first_token_time = None
        self.prefill_time = None
        self.enterance_time = None
        self.exit_time = None

    def put(self, token):
        self.enterance_time = time.perf_counter()
        if len(self.token_generation_times) == 0:
            # Capture the prefill time
            prefill_done_time = time.perf_counter()
            super().put(token)
            first_token_done_time = time.perf_counter()
            self.first_token_time = (first_token_done_time - prefill_done_time) * 1000 # millisecond
            self.prefill_time = (prefill_done_time - self.start_generation_time) * 1000 # millisecond
            self.token_generation_times.append(first_token_done_time)
            self.exit_time = time.perf_counter()
        else:
            # Capture the token generation time
            super().put(token)
            token_done_time = time.perf_counter()
            previous_token_done_time = self.token_generation_times.pop()
            self.token_generation_times.append((token_done_time - previous_token_done_time) * 1000) # millisecond
            self.token_generation_times.append(token_done_time)
            # Capture the inter-token time
            self.inter_token_times.append((self.enterance_time - self.exit_time) * 1000) # millisecond
            self.exit_time = time.perf_counter()



