#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import get_full_repo_name

from sal.utils.hub import get_dataset_revisions


@dataclass
class Config:
    approach: Literal["best_of_n", "beam_search", "dvts"] = "best_of_n"
    model_path: str = "ibm-granite/granite-3.1-2b-instruct"
    gpu_memory_utilization: float = (
        0.5  # vllm is allocated 0.5 of GPU memory, the PRM uses the rest
    )
    prm_path: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    # Output Related Options
    output_dir: str = None
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    hub_dataset_private: bool = False
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    dataset_name_evidence: str = "/home/MattiaTritto_STD/llm-project/src/sql_dataset_evidence"
    dataset_name_no_evidence: str = "/home/MattiaTritto_STD/llm-project/src/sql_dataset_no_evidence"
    working_dir: str =  "/llm-project/search-and-learn/"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    # num_samples: int = None
    num_samples: int = None
    eval_file_evidence: str = "/home/MattiaTritto_STD/llm-project/evaluation/results/evidence/predict.json"
    eval_file_no_evidence: str = "/home/MattiaTritto_STD/llm-project/evaluation/results/no_evidence/predict.json"

    # Chat template related options
    system_prompt: str = "You are a Text-to-SQL model assistant."
    custom_chat_template: str = """{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = "Knowledge Cutoff Date: April 2024.
Today's Date: " + strftime_now('%B %d, %Y') + ".
You are Granite, developed by IBM." %}
    {%- if tools and documents %}
        {%- set system_message = system_message + " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.

Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}
    {%- elif tools %}
        {%- set system_message = system_message + " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request." %}
    {%- elif documents %}
        {%- set system_message = system_message + " Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." %}
    {%- else %}
        {%- set system_message = system_message + " You are a helpful AI assistant." %}
    {%- endif %}
    {%- if 'citations' in controls and documents %}
        {%- set system_message = system_message + '

In your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}
    {%- endif %}
    {%- if 'hallucinations' in controls and documents %}
        {%- set system_message = system_message + '

Finally, after the response is written, include a numbered list of sentences from the response that are potentially hallucinated and not based in the documents.' %}
    {%- endif %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>
' }}
{%- if tools %}
    {{- '<|start_of_role|>tools<|end_of_role|>' }}
    {{- tools | tojson(indent=4) }}
    {{- '<|end_of_text|>
' }}
{%- endif %}
{%- if documents %}
    {{- '<|start_of_role|>documents<|end_of_role|>' }}
    {%- for document in documents %}
        {{- 'Document ' + loop.index0 | string + '
' }}
        {{- document['text'] }}
        {%- if not loop.last %}
            {{- '

'}}
        {%- endif%}
    {%- endfor %}
    {{- '<|end_of_text|>
' }}
{%- endif %}
{%- for message in loop_messages %}
    {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>
' }}
    {%- if loop.last and add_generation_prompt %}
        {{- '<|start_of_role|>assistant' }}
            {%- if controls %}
                {{- ' ' + controls | tojson()}}
            {%- endif %}
        {{- '<|end_of_role|>' }}
    {%- endif %}
{%- endfor %}
"""
    # Search Related Options
    n: int = 1
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 1
    seed: int = 42
    # max_tokens: int = 2048
    max_tokens: int = 2048
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    # DVTS / Beam Search options
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40
    lookahead: int = 1

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False

    def __post_init__(self):
        if self.approach == "dvts":
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.n // self.beam_width

        if self.approach == "beam_search":
            # TODO: implemented a batched version
            if self.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for beam_search")

        # Setting up push to hub dataset
        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                # Set default based on model name. We prepend the username for compatibility with the repo checks below.
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            if self.approach == "beam_search" or self.approach == "dvts":
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--m-{self.beam_width}--iters-{self.num_iterations}--look-{self.lookahead}--seed-{self.seed}--agg_strategy--{self.agg_strategy}"
            elif self.approach == "best_of_n":
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--seed-{self.seed}--agg_strategy-{self.agg_strategy}"
            else:
                raise ValueError(f"Unknown approach {self.approach}")
            if self.dataset_start is not None and self.dataset_end is not None:
                self.revision = (
                    f"{self.revision}--chunk-{self.dataset_start}_{self.dataset_end}"
                )

            # Early exit if the revision on the Hub already exists
            if not self.overwrite_hub_revision and self.revision in revisions:
                # logger.info(f"Revision {revision} already exists on the Hub. Exiting.")
                exit()
