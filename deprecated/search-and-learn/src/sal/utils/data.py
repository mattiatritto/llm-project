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



import logging
import time
from pathlib import Path
import json
import re

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config, evidence: bool) -> Dataset:
    # dataset = load_dataset(config.dataset_name, split=config.dataset_split) # MODIFIED

    if evidence:
        dataset = load_from_disk(config.dataset_name_evidence)
    else: 
        dataset = load_from_disk(config.dataset_name_no_evidence)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            config.output_dir = f"data/{config.model_path}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        dataset.to_json(
            f"{config.output_dir}/{config.approach}_completions.jsonl", lines=True
        )
        logger.info(
            f"Saved completions to {config.output_dir}/{config.approach}_completions.jsonl"
        )



def extract_sql_query(sql_query):
    if not isinstance(sql_query, str):
        return None
        
    match = re.search(r'\[SQL\](.*?)\[/SQL\]', sql_query, re.DOTALL)
    return match.group(1) if match else sql_query


# ADDED
def save_dataset_for_eval(dataset: any, output_path: str) -> None:
    """
    Save the dataset to a JSON file with a simpler format using 'progressive_number' as the key.

    Args:
        dataset: The processed dataset with features including 'pred' and 'db_id'.
        output_path: Path to save the JSON file (e.g., 'output_simple.json').
    """

    # Initialize the dictionary for the custom format
    simple_data = {}

    # Iterate over the dataset rows
    for idx, item in enumerate(dataset):
        # Extract the SQL query from the 'pred' field
        sql_query = item["pred"].strip()
        sql_query = extract_sql_query(sql_query)

        # Extract database ID for metadata (fallback to 'unknown_db' if missing)
        db_id = item.get("db_id", "unknown_db")

        # Construct the value string
        value = f"{sql_query}\t----- bird -----\t{db_id}"



        simple_data[f"{idx}"] = value

    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved dataset in simple JSON format to {output_path}")