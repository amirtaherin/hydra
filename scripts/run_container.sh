#!/usr/bin/env bash
# Copyright (c) 2026 Amir Taherin
# Licensed under the MIT License (see LICENSE).
jetson-containers run --privileged --env HUGGINGFACE_TOKEN=hf_<replace with your token> \
	-v ../:/home/hydra/ \
	-v /usr/bin:/usr/bin \
	-v /proc:/proc \
	-v /sys:/sys \
	$(autotag nano_llm) \
	bash -c "pip install --upgrade transformers && cd /home/amir/ && /bin/bash"
