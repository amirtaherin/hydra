# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
import json

class PromptProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.total_num_prompts = 0

    def __enter__(self):
        """Opens the file when entering the context manager."""
        self.file = open(self.filepath, 'r')

        # Count the total number of prompts in the file
        self.total_num_prompts = sum(1 for _ in self.file)

        # Reset the file pointer to the beginning of the file
        self.file.seek(0)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the file when exiting the context manager."""
        if self.file:
            self.file.close()

    def get_next_line(self):
        """Returns the next line from the JSONL file."""
        if self.file:
            line = self.file.readline()
            if line:
                return json.loads(line.strip())  # Parse the JSON and remove the newline character
            return None
        raise ValueError("File not opened. Please use the context manager to open the file.")

    def get_total_num_prompts(self):
        """Returns the total number of prompts in the JSONL file."""
        return self.total_num_prompts

    def get_next_prompt(self):
        """Returns the next prompt from the JSONL file."""
        line = self.get_next_line()
        if line:
            return line['prompt']
        return None

# Usage example
if __name__ == "__main__":
    with PromptProcessor('./input_data.jsonl') as reader:
        print(f"Total number of prompts: {reader.get_total_num_prompts()}")
        prompt = reader.get_next_prompt()
        while prompt is not None:
            print(prompt)
            a= input("Press Enter to continue...")
            prompt = reader.get_next_prompt()

