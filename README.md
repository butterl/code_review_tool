# code_review_tool
use openai API to review code in repo

```

usage: code_reviewer.py [-h] [--model_name MODEL_NAME] [--language LANGUAGE] [--output_dir OUTPUT_DIR] [--chunk_size CHUNK_SIZE] repo_path

Code Review Tool

positional arguments:
  repo_path             Path to the code repository

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        OpenAI model name
  --language LANGUAGE   Programming language (e.g. CPP, PYTHON, JS)
  --output_dir OUTPUT_DIR
                        Directory for Markdown output
  --chunk_size CHUNK_SIZE
                        Chunk size for text splitter
```

# example
```
python3 ./code_reviewer.py --model_name gpt-3.5-turbo-0125 --language python ./
```
