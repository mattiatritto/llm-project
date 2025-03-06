# llm-project

## Overview
This project focuses on fine-tuning and utilizing test-time scaling techniques to obtain a small open-source Large Language Model (LLM) optimized for Text-to-SQL tasks.

## Installation

Follow these steps to set up and run the project:

1. **Create the project directory:**
   ```sh
   mkdir llm-project
   ```
2. **Clone the repository:**
   ```sh
   git clone https://github.com/mattiatritto/llm-project
   ```
3. **Navigate to the project directory:**
   ```sh
   cd llm-project
   ```
4. **Build the Apptainer container:**
   ```sh
   apptainer build --fakeroot Apptainer.sif Apptainer.def
   ```
5. **Run the Apptainer container with necessary bindings:**
   ```sh
   apptainer run --nv --bind $(pwd):/llm-project,/storage/shared_cache/shared_huggingface:/storage/shared_cache/shared_huggingface Apptainer.sif
   ```

## Usage

1. **Run the container:**
   ```sh
   apptainer run --nv --bind $(pwd):/llm-project,/storage/shared_cache/shared_huggingface:/storage/shared_cache/shared_huggingface Apptainer.sif
   ```
2. **Edit the configuration file:**
   ```sh
   nano config.yaml
   ```
3. **Run inference:**
   ```sh
   python $INFERENCE
   ```
4. **View results:**
   ```sh
   cat $EV
   ```
   or
   ```sh
   cat $NO_EV
   ```
5. **Run evaluation:**
   ```sh
   $EVALUATION
   ```

## Contributing
Feel free to open issues or submit pull requests to improve the project.

---
For any further queries, please refer to the project repository or contact the maintainers.

