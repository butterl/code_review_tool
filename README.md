# code_review_tool
use openai API to review code in repo

```
usage: code_reviewer.py [-h] [--model_name MODEL_NAME] [--language LANGUAGE] [--output_dir OUTPUT_DIR] [--chunk_size CHUNK_SIZE] [--estimate_cost ESTIMATE_COST] repo_path

Code Review Tool

positional arguments:
  repo_path             Path to the code repository

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        OpenAI model name
  --language LANGUAGE   Programming language (e.g. cpp, python, js)
  --output_dir OUTPUT_DIR
                        Directory for Markdown output
  --chunk_size CHUNK_SIZE
                        Chunk size for text splitter
  --estimate_cost ESTIMATE_COST
                        Run check only to see maybe cost
```

# example
```
estimate token and cost
python3 ./code_reviewer.py --model_name gpt-3.5-turbo-0125 --language cpp --chunk_size 9500 --estimate_cost True ../${repo_to_review}

check code
python3 ./code_reviewer.py --model_name gpt-3.5-turbo-0125 --language cpp --chunk_size 9500 ../${repo_to_review}
```
