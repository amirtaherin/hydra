# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
import argparse
from inference.hf_profiler.llm_inference import LLMInference

def parse_opt():
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument('--input', type=str, default='input.jsonl', help='Input file path')
    # Model
    parser.add_argument('--model-names', type=str, default='inputs/models/models.json', help='Model names file path')
    # Data type
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='Data type for inference')
    # max new tokens
    parser.add_argument('--max-new-tokens', type=int, default=500, help='Maximum number of new tokens to generate')
    # max prompts
    parser.add_argument('--max-prompts', type=int, default=-1, help='Maximum number of prompts to process (-1 for all)')
    # Download only
    parser.add_argument('--download-only', action='store_true', help='Download models without running inference')
    # Output directory
    parser.add_argument('--result-dir', type=str, default='results', help='Result directory')
    # Inference info => deprecated
    # parser.add_argument('--info-path', type=str, default='info_results.csv', help='Inference info file path')
    # Tegrastat path => deprecated
    # parser.add_argument('--tgs-path', type=str, default='tgs_results.csv', help='Tegrastat file path')

    return parser.parse_args()

def main(opt):
    # Initialize LLMInference
    llm = LLMInference(opt)
    llm.inference_engine()

if __name__ == '__main__':
    # Parse command line options
    opt = parse_opt()
    # Run main
    main(opt)

