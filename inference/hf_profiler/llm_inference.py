# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
import pathlib as pl
import json, csv
from inference.hf_profiler.process_prompt import PromptProcessor
import socket
_hostname = socket.gethostname().lower()
if 'thor' in _hostname:
    from inference.telemetry.tegrastats_jp71_thor import Tegrastats
else:
    from inference.telemetry.tegrastats import Tegrastats
import torch
import time
import gc
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "true"

class LLMInference:
    def __init__(self, opt):
        self.input = pl.Path(opt.input)
        self.model_names = pl.Path(opt.model_names)
        self.result_dir = pl.Path(opt.result_dir)
        self.__finalize_path()
        self.dtype = self.__set_dtype(opt)
        self.max_new_tokens = opt.max_new_tokens
        self.max_prompts = opt.max_prompts
        self.download_only = opt.download_only
        self.model_names = self.__json_iterator(self.model_names)
        self.device = self.__set_device()
        self.total_num_prompt = 0 if self.download_only else self.__get_num_prompts()
        self.total_parameters = None
        self.total_bytes = None
        self.model_type = None
        self.print()

    def download_models(self):
        """Download all models and tokenizers without running inference."""
        from huggingface_hub import snapshot_download
        for model in self.model_names:
            model_name, model_path = model['name'], model['hf_path']
            print(f'Downloading model: {model_name} ({model_path})...')
            try:
                snapshot_download(model_path)
                print(f'  Done: {model_name}')
            except Exception as e:
                print(f'  Failed: {model_name}: {e}')
        print("All downloads complete.")

    def inference_engine(self):
        """Inference on the models and prompts and profile them"""
        if self.download_only:
            self.download_models()
            return

        for model in self.model_names:
            model_name, model_path = model['name'], model['hf_path']
            print(f'Inference on model: {model_name}, path: {model_path}')

            # Start the tegrastats profiler
            # Note: The tegrastats profiler is only available on NVIDIA GPUs
            # Note: threads should be spawned before tokenizer due to conflict with multiprocessing
            tgs = Tegrastats(log_file=f"{self.result_dir}/TGS_{model_name}_{self.dtype}_.csv", interval=10)
            tgs_p = tgs.start_tegrastats()                      # Start the tegrastats and get the process instead


            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path,device=self.device)
            #model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
            model = self._load_model(model_path, dtype=self.dtype)
            if model is None:
                tgs.stop_tegrastats(tgs_p)
                raise SystemExit(
                    f"Model load failed for {model_name} - see the "
                    "traceback above (check torch/transformers versions "
                    "match your platform; see README).")
            model.eval()

            # Get the number of parameters and size of model
            self.total_parameters = sum(p.numel() for p in model.parameters())
            self.total_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
            self.model_type = (next(model.parameters())).dtype
            #print(f'Total parameters: {self.total_parameters}')
            #print(f'Total Memory (MB): {self.total_bytes/1e6:.2f}')
            #print(f'Model type: {self.model_type}')

            # Process prompts and run inference

            # Run inference
            self.run_inference(model, tokenizer, model_name)


            # Stop the tegrastats profiler
            tgs.stop_tegrastats(tgs_p)                     # Stop the tegrastats profiler


            # clean the memory
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache() # clear the cache of GPU
            torch.cuda.synchronize() # Ensure all GPU operations are done
        print("Done")



    def run_inference(self, model, tokenizer, model_name):
        with open(f"{self.result_dir}/INFO_{model_name}_{self.dtype}_.csv", "w", newline='') as csvfile:
            header = [
                'prompt_number',
                'start_time',
                'end_time',
                'model_name',
                'size_of_model',
                'tokenization_time',
                'prefill_time',
                'time_to_first_token',
                'mean_token_generation_time',
                'variance_token_generation_time',
                'mean_token_decode_time',
                'variance_token_decode_time',
                'mean_inter_token_latency',
                'variance_inter_token_latency',
                'end_to_end_latency',
                'estimated_end_to_end_latency',
                'output_length',
                'total_generation_time',
                'total_number_of_prompts',
                'total_number_of_tokens',
                'total_number_of_tokens_in_input',
                'total_number_of_tokens_in_output',
                'total_number_of_tokens_in_input_plus_output',
                'prompt_tokens_per_sec',
                'generation_tokens_per_sec'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

            #  ---- INFERENCE ----
            counter = 0
            with PromptProcessor(self.input) as prompt_processor:
                prompt = prompt_processor.get_next_prompt() # get the next prompt
                counter += 1                                # increment the prompt counter
                print(f"Processing prompt {counter}...")    # print the prompt number

                # --- INFERENCE LOOP ---
                while prompt and (self.max_prompts < 0 or counter <= self.max_prompts):
                    # HIGH RESOLUTION TIMER — wall-clock ns at prompt start.
                    # MUST be inside the while loop so each prompt gets its own
                    # start_time. Was previously above the loop, which made all
                    # rows share the same start_time and broke per-prompt slicing
                    # in data_unifier_parallel.py.
                    START = time.time_ns()

                    # Tokenization timing
                    t0 = time.perf_counter()                # Start of tokenization
                    if self.device == 'cuda:0':
                        torch.cuda.synchronize()
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    if self.device == 'cuda:0':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()                # End of tokenization
                    tokenization_time = t1 - t0             # Record the tokenization time

                    input_ids = inputs["input_ids"]
                    attention_mask = inputs.get("attention_mask", None)

                    # Prefill timing
                    with torch.no_grad():
                        if self.device == 'cuda:0':
                            torch.cuda.synchronize()
                        t2 = time.perf_counter()            # Start of prefill
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                        if self.device == 'cuda:0':
                            torch.cuda.synchronize()
                        t3 = time.perf_counter()            # End of prefill
                        prefill_time = t3 - t2              # Record the prefill time
                        past_key_values = \
                            outputs.past_key_values         # Cache for efficient decoding

                    # Generation timing
                    generated = input_ids
                    token_generation_times = []
                    inter_token_latencies = []
                    token_decode_times = []

                    generation_start_times = []             # Start times of each token
                    generation_end_times = []               # End times of each token

                    decode_end = None                       # End time of decoding
                    first_token = True                      # First token uses prefill logits


                    # Token-by-token generation with profiling
                    for i in range(self.max_new_tokens):
                        # --- MODEL INFERENCE: TOKEN GENERATION TIME ---
                        with torch.no_grad():
                            if self.device == 'cuda:0':
                                torch.cuda.synchronize()
                            t_start = time.perf_counter()           # Start of token generation for token i

                            # Skip forward pass on first iteration — logits are
                            # already available from the prefill outputs.
                            # This matches llama.cpp profiler behavior and avoids
                            # a redundant model.forward() call.
                            if not first_token:
                                outputs = model(                    # model.forward() method
                                    input_ids=generated[:, -1:],    # Only pass the last token
                                    past_key_values=past_key_values,
                                    use_cache=True                  # Use cache for efficient decoding
                                )

                                if self.device == 'cuda:0':
                                    torch.cuda.synchronize()

                            # Select the next token with greedy approach
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                            generated = torch.cat([generated, next_token], dim=-1)
                            past_key_values = outputs.past_key_values  # Update the cache

                            t_end = time.perf_counter()             # End of token generation for token i

                        # --- DECODE TIME (one token at a time) ---
                        # Decode the generated token
                        decoded_token = tokenizer.decode(next_token[0], skip_special_tokens=True)
                        decode_end = time.perf_counter()              # End of decoding

                        # --- TOKEN TIMING ---
                        # Time each token and append to the list
                        token_time = t_end - t_start                # Record the time taken to generate the token i
                        decode_time = decode_end - t_end            # Record the time taken to decode the token i
                        token_generation_times.append(token_time)   # Append the time to the list
                        generation_start_times.append(t_start)      # Append the start time of token i
                        generation_end_times.append(t_end)          # Append the end time of token i
                        token_decode_times.append(decode_time)      # Record the decode time
                        first_token = False
                    
                    # --- END OF TOKEN GENERATION ---

                    # --- HIGH RESOLUTION TIMER ---
                    END = time.time_ns()                        # End time of the prompt processing pipeline


                    # Inter-token latency
                    # Calculate the inter-token latencies between generation of each two consecutive tokens
                    inter_token_latencies = [
                        generation_end_times[i+1] - generation_end_times[i]
                        for i in range(len(generation_start_times) - 1)
                    ]

                    # Total generation + decoding time
                    total_gen_and_dec_time = \
                        decode_end - generation_start_times[0]  # Total time taken to generate all tokens

                    # Time to first token
                    # TTFT = tokenization + prefill + generation + decode time of the first token
                    time_to_first_token = \
                        tokenization_time + prefill_time + token_generation_times[0] + token_decode_times[0]

                    # End-to-end latency

                    # E2E = tokenization + prefill + generation + decode time of the last token
                    e2e_latency = decode_end - t0

                    # estimated e2e latency shows the timing overheads
                    estimated_e2e_latency = tokenization_time + prefill_time + total_gen_and_dec_time

                    # Output length
                    output_length = len(generated[0]) - len(input_ids[0])  # Number of tokens generated

                    # Tokens per second
                    prompt_tokens_per_sec = len(input_ids[0]) / prefill_time if prefill_time > 0 else 0
                    generation_tokens_per_sec = output_length / total_gen_and_dec_time if total_gen_and_dec_time > 0 else 0

                    # Write the results to the CSV file
                    # Note: time values are stored in seconds (*1000) to get ms
                    # Note: start and end are in nano_second
                    writer.writerow({
                        'prompt_number': counter,
                        'start_time': START,
                        'end_time': END,
                        'model_name': model_name,
                        'size_of_model': self.total_bytes,
                        'tokenization_time': tokenization_time,
                        'prefill_time': prefill_time,
                        'time_to_first_token': time_to_first_token,
                        'mean_token_generation_time': np.mean(token_generation_times),
                        'variance_token_generation_time': np.var(token_generation_times),
                        'mean_token_decode_time': np.mean(token_decode_times),
                        'variance_token_decode_time': np.var(token_decode_times),
                        'mean_inter_token_latency': np.mean(inter_token_latencies),
                        'variance_inter_token_latency': np.var(inter_token_latencies),
                        'end_to_end_latency': e2e_latency,
                        'estimated_end_to_end_latency': estimated_e2e_latency,
                        'output_length': output_length,
                        'total_generation_time': total_gen_and_dec_time,
                        'total_number_of_prompts': self.total_num_prompt,
                        'total_number_of_tokens': len(generated[0]),
                        'total_number_of_tokens_in_input': len(input_ids[0]),
                        'total_number_of_tokens_in_output': output_length,
                        'total_number_of_tokens_in_input_plus_output': len(generated[0]),
                        'prompt_tokens_per_sec': prompt_tokens_per_sec,
                        'generation_tokens_per_sec': generation_tokens_per_sec
                    })
                    
                    # Get the next prompt
                    prompt = prompt_processor.get_next_prompt() # get the next prompt
                    counter += 1                                # increment the prompt counter
                    print(f"Processing prompt {counter}...")    # print the prompt number


                    # Print the results
                    """
                    print(f"Model: {model_name}")
                    print(f"Prompt: {prompt}")
                    print(f"Tokenization time: {tokenization_time * 1000:.4f} milliseconds")
                    print(f"Prefill time: {prefill_time * 1000:.4f} milliseconds")
                    print(f"Time to first token: {time_to_first_token * 1000:.4f} milliseconds")
                    print(f"Mean Token generation times: {np.mean(token_generation_times) * 1000:.4f} milliseconds")
                    print(f"Variance Token generation times: {np.var(token_generation_times) * (1000**2):.4f} milliseconds^2")
                    print(f"Mean Token decode times: {np.mean(token_decode_times) * 1000:.4f} milliseconds")
                    print(f"Variance Token decode times: {np.var(token_decode_times) * (1000**2):.4f} milliseconds^2")
                    print(f"Mean Inter-token latencies: {np.mean(inter_token_latencies) * 1000:.4f} milliseconds")
                    print(f"Variance Inter-token latencies: {np.var(inter_token_latencies) * (1000**2):.4f} milliseconds^2")
                    print(f"End-to-end latency: {e2e_latency * 1000:.4f} milliseconds")
                    print(f"Estimated end-to-end latency: {estimated_e2e_latency * 1000:.4f} milliseconds")
                    print(f"Output length: {output_length} tokens")
                    print(f"Total generation time: {total_gen_and_dec_time * 1000:.4f} milliseconds")
                    print(f"Total number of prompts: {self.total_num_prompt}")
                    print(f"Total number of tokens: {len(generated[0])}")
                    print(f"Total number of tokens in the input: {len(input_ids[0])}")
                    print(f"Total number of tokens in the output: {output_length}")
                    print(f"Total number of tokens in the input + output: {len(generated[0])}")
                    """

                    








    def _load_model(self, model_name, dtype):
        """Load the model and handle OOM errors gracefully."""
        try:
            print(f"Trying to load {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=0, #"auto"          # instead of .to("cuda:0")
                torch_dtype=dtype,  # or bfloat16 / leave out if unsure
            )
            print(f"✅ Successfully loaded: {model_name}")
            return model
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ OOM when loading {model_name}: {e}")
            # Clean up
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared.")
            return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Unexpected error: {e}")
            torch.cuda.empty_cache()
            return None

    def print(self):
        for key, value in self.__dict__.items():
            print(f'{key}: {value}')

    def __finalize_path(self):
        """Finalize the path to the input, result directory, info and tgs paths"""
        file = pl.Path(__file__).resolve()
        root = file.parents[1] # root directory of the project
        self.input = root / self.input
        self.model_names = root / self.model_names
        self.result_dir = root / self.result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def __set_device(self):
        """Set device to cuda if available"""
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __set_dtype(self, opt):
        """Set dtype to float16 and bfloat16 if cuda is available else float32, based on the option"""
        if opt.dtype == 'float16' and torch.cuda.is_available():
            return torch.float16
        elif opt.dtype == 'bfloat16' and torch.cuda.is_available():
            return torch.bfloat16
        else:
            return torch.float32

    def __json_iterator(self, json_file):
        """Generator for model names"""
        with open(json_file, 'r') as file:
            data = json.load(file)
            for model in data['models']:
                yield model   # returns a generator and one model at a time


    def __get_num_prompts(self):
        """Get the number of prompts in the input file"""
        with PromptProcessor(self.input) as prompt_processor:
            return prompt_processor.get_total_num_prompts() # number of prompts in the input file
