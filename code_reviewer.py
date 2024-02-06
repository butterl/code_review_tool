#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Code Review Tool

This script is designed to automate the process of reviewing code by utilizing OpenAI's language models. 
It scans a given repository, filters files based on language and .gitignore patterns, and then processes each file to generate a review report. 
The review focuses on various aspects such as performance, security, maintainability, readability, scalability, and resource management.

Version: 1.0
License: [GPLv3]
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import fnmatch
import shutil
import concurrent.futures
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI


# Fake run to see estimate cost
ONLY_CHECK_COST = False

# Model rates (per 1000 tokens)
MODEL_RATES = {
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03, "chunk_size":8000},
    "gpt-4-1106-preview": {"input": 0.01, "output": 0.03, "chunk_size":8000},
    "gpt-4-1106-vision-preview": {"input": 0.01, "output": 0.03, "chunk_size":8000},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03, "chunk_size":8000},
    "gpt-4": {"input": 0.03, "output": 0.06, "chunk_size":8000},
    "gpt-4-32k": {"input": 0.06, "output": 0.12, "chunk_size":32000},
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0010, "chunk_size":4000},
    "gpt-3.5-turbo-1106": {"input": 0.0010, "output": 0.0020, "chunk_size":4000},
    "gpt-3.5-turbo": {"input": 0.0010, "output": 0.0020, "chunk_size":4000},
    "gpt-3.5-16K": {"input": 0.0030, "output": 0.0060, "chunk_size":4000},
    # Add rates for other models
}

SYSTEM_PROMPT = '''
您作为一名经验丰富的软件架构师，被邀请对一份代码进行深入审查.请基于您的专业知识和经验，
关注以下领域并提出具体的改进意见：性能、安全性、稳定性.
期待您的反馈能具体、针对性强、可直接指导后续的代码优化(比如给出优化后代码).
We will tip you $10 for a perfect answer.

审查要点
1.性能：请识别出具体影响性能的代码行或部分,提供改善性能的详细建议
2.安全性：指出潜在的安全漏洞,并给出具体的解决方案以增强代码的安全性
3.稳定性：找出可能引起业务稳定性问题的代码段,并提出具体的改进措施

反馈格式要求
- 请以标准Markdown形式提供反馈
- 每种类型的问题不限制个数,如果某个类型问题不存在则不用返回该行
- 每个类型的问题最后添加markdown 分割符(---)以及换行符(\n)
格式如下:
\n### 问题分类:(性能/安全/稳定 etc.)
### 问题位置:
(提供问题所在的函数名或模块名及其上下文的5-10行代码.并简要描述问题所在的上下文)
### 问题描述:
(指明问题原因和影响范围和修改思路)
### 修改示例:
(请提供基于修改思路具体、可操作的修改建议及关键修改点示例)

注意事项
- 请确保您的反馈专注与审查要点，对整体代码质量的提升
- 您可能拿到的是代码片段,请假设代码已可以编译运行
- 请使用中文回答,并确保回复格式准确无误
'''

def merge_feedback_and_generate_markdown(feedback_data):
    """
    Merges feedback from multiple threads and generates a single Markdown table,
    ensuring robust handling of JSON structure variations.
    
    Parameters:
    - feedback_data: A list of JSON structured feedback strings from multiple threads.
    
    Returns:
    - A string containing merged feedback in a single Markdown table format.
    """
    markdown_output = "## 代码审查反馈汇总\n\n"
    markdown_output += "| 问题分类 | 问题位置 | 问题描述 | 修改建议 |\n"
    markdown_output += "|------|------|----------|------|\n"

    # Assuming feedback_data is directly the dictionary containing the 'feedback' list
    for feedback in feedback_data['feedback']:
        area = feedback['area']  # Access 'area' from each feedback item
        for issue in feedback['issues']:
            location = issue['loc']
            description = issue['desc']
            suggestion = issue['sug']
            markdown_output += f"| {area} | {location} | {description} | {suggestion} |\n"

    return markdown_output


# Language and file extension mapping
LANGUAGE_FILE_EXTENSIONS = {
    "CPP": ['.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.c', '.h'],
    "PYTHON": ['.py'],
    "JS": ['.js', '.ts', '.tsx']
    # Add mappings for other languages and extensions
}

DEFAULT_BASE_URL = "https://api.openai.com"


def get_api_key():
    """
    Retrieves the OpenAI API key from environment variables.

    Returns:
    - str: The API key.

    Raises:
    - SystemExit: If the API key environment variable is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not set.")
        sys.exit()
    return api_key


def get_base_url():
    """
    Retrieves the base URL for the OpenAI API from environment variables, defaulting to a predefined URL if not set.

    Returns:
    - str: The base URL for the OpenAI API.

    Raises:
    - SystemExit: If the base URL environment variable is not set.
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        logging.error("OPENAI_API_KEY environment variable not set.")
        base_url = DEFAULT_BASE_URL
    else:
        if not base_url.endswith("/"):
            base_url += "/"
    base_url += "v1"
    return base_url


client = OpenAI(
    api_key=get_api_key(),
    base_url=get_base_url(),
)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize TikToken
tokenizer = tiktoken.encoding_for_model("gpt-4")


def recursive_scandir(path, gitignore_patterns):
    """
    Recursively scans a directory, ignoring specified directories and files that match patterns in .gitignore.

    Parameters:
    - path (str): The directory path to scan.
    - gitignore_patterns (list of str): Patterns to ignore, typically sourced from .gitignore.

    Returns:
    - generator: Yields Path objects for files that don't match the ignore patterns and aren't in ignored directories.
    """
    ignore_dirs = {'.git', 'fuzztest', 'unittest', 'tests', 'test', 'third_party'}
    for entry in os.scandir(path):
        if entry.is_dir():
            if entry.name in ignore_dirs:
                continue
            yield from recursive_scandir(entry.path, gitignore_patterns)
        else:
            file = Path(entry.path)
            if not any(fnmatch.fnmatch(str(file), pattern) for pattern in gitignore_patterns):
                yield file


def scan_files(repo_path, gitignore_patterns):
    """
    Scans files in a repository, applying .gitignore patterns.

    Parameters:
    - repo_path (str): The path to the repository to scan.
    - gitignore_patterns (list of str): Patterns from .gitignore to exclude files.

    Returns:
    - list: A list of Path objects representing the files to review.
    """
    files_to_review = []
    for file in recursive_scandir(repo_path, gitignore_patterns):
        if 'test' in file.name or 'mock' in file.name:
            continue
        files_to_review.append(file)
    return files_to_review


def filter_files(files, file_extensions):
    """
    Filters files by their extension.

    Parameters:
    - files (list of Path objects): Files to be filtered.
    - file_extensions (list of str): Extensions to include in the filter.

    Returns:
    - list: A list of Path objects that match the specified extensions.
    """
    filtered_files = []
    for file in files:
        if any(file.name.endswith(ext) for ext in file_extensions):
            filtered_files.append(file)
    return filtered_files


def process_repository(repo_path, file_extensions):
    """
    Processes a repository to identify files for review based on file extensions and .gitignore patterns.

    Parameters:
    - repo_path (str): Path to the repository.
    - file_extensions (list of str): List of file extensions to include in the review.

    Returns:
    - list: A list of strings representing file paths to be reviewed.

    Raises:
    - ValueError: If repo_path is not a valid directory.
    """
    if not os.path.isdir(repo_path):
        raise ValueError("Invalid repo_path. It must be a valid directory.")

    gitignore_patterns = []  # Initialize gitignore_patterns as an empty list
    gitignore_path = os.path.join(repo_path, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as gitignore_file:
            gitignore_patterns.extend(line.strip() for line in gitignore_file if line.strip() and not line.startswith('#'))

    files_to_review = scan_files(repo_path, gitignore_patterns)
    filtered_files = filter_files(files_to_review, file_extensions)

    logging.info(f"Total files identified for review: {len(filtered_files)}")
    return [str(file) for file in filtered_files]


def get_chunk_size(model_name):
    """
    Retrieves the chunk size for a given model from the MODEL_RATES dictionary.

    Parameters:
    - model_name (str): The name of the model.

    Returns:
    - int: The chunk size associated with the model or a default value if the model is not found.
    """
    data = MODEL_RATES.get(model_name, {"input": 0, "output": 0, "chunk_size": 4000})
    if data:
        return data["chunk_size"]
    return 4000


def estimate_cost(tokens_in, token_out, model_name):
    """
    Estimates the cost of processing given the number of input and output tokens and the model used.

    Parameters:
    - tokens_in (int): The number of input tokens.
    - token_out (int): The number of output tokens.
    - model_name (str): The model name to use for rate lookup.

    Returns:
    - float: The estimated cost for the operation.
    """
    rates = MODEL_RATES.get(model_name, {"input": 0, "output": 0, "chunk_size": 0})
    return (tokens_in / 1000) * (rates["input"]) + (token_out / 1000)*(rates["output"])


def remove_license_header(content, header_end_identifiers, header_search_limit=30):
    """
    Removes license headers from file content based on header end identifiers.

    Parameters:
    - content (str): The content of the file.
    - header_end_identifiers (list of str): Identifiers that denote the end of a header.
    - header_search_limit (int): The maximum number of lines to search for the header end identifiers.

    Returns:
    - str: The content with the license header removed, if found.
    """
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if any(identifier in line for identifier in header_end_identifiers):
            # remove the header
            return '\n'.join(lines[index + 1:])
        if index >= header_search_limit - 1:
            break
    return content # find no header


def process_code_chunk(doc, model_name, max_tokens, system_prompt, total_cost):
    """
    Processes a chunk of code using OpenAI's API to generate a review.

    Parameters:
    - doc (Document): The document object representing a chunk of code.
    - model_name (str): The OpenAI model to use.
    - max_tokens (int): The maximum number of tokens for the OpenAI API call.
    - system_prompt (str): The system prompt for the review.
    - total_cost (float): The cumulative cost of processing up to this chunk.

    Returns:
    - tuple: A tuple containing the new total cost and the review content for the chunk.
    """
    tokens_in = len(tokenizer.encode(system_prompt + doc.page_content))

    if ONLY_CHECK_COST: #jump real api call to estimate cost for large content
        tokens_out = 1000
        estimated_cost = estimate_cost(tokens_in, tokens_out, model_name)
        new_total_cost = total_cost + estimated_cost
        logging.info(f"tokens_in: {tokens_in}, len_string:{len(system_prompt + doc.page_content)}")
        return (new_total_cost, '')

    completion = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": doc.page_content}
        ]
    )
    result_content = completion.choices[0].message.content


    tokens_out = len(tokenizer.encode(result_content))

    estimated_cost = estimate_cost(tokens_in, tokens_out, model_name)
    new_total_cost = total_cost + estimated_cost
    logging.info(f"tokens_in: {tokens_in}, len_string:{len(system_prompt + doc.page_content)}; tokens_out:{tokens_out},len_string:{len(result_content)}")
    return (new_total_cost, result_content)  # Indicate continuation.


def review_code_with_openai(file_path, model_name, output_dir, language, chunk_size, max_tokens=2000, total_cost=0):
    """
    Reviews code in a given file using OpenAI, writing the review to a Markdown file.

    Parameters:
    - file_path (str): Path to the file to review.
    - model_name (str): The OpenAI model to use.
    - output_dir (str): Directory where the Markdown output will be saved.
    - language (str): The programming language of the file.
    - max_tokens (int, optional): The maximum number of tokens for the OpenAI API call.
    - total_cost (float, optional): The initial total cost before processing this file.
    - chunk_size (int, optional): The size of text chunks to process individually.

    Returns:
    - float: The updated total cost after processing this file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cleaned_content = remove_license_header(content, ["*/", "#", "//"])
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return
    logging.info(f"Reviewing file {file_path}")

    splitter = RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=chunk_size, chunk_overlap=0)
    docs = splitter.create_documents([cleaned_content])
    output_path = Path(output_dir) / f"{Path(file_path)}_review.md"

    collected_reviews = []
    total_cost = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_doc = {executor.submit(process_code_chunk, doc, model_name, max_tokens, SYSTEM_PROMPT, total_cost): doc for doc in docs}
        for future in concurrent.futures.as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                new_total_cost, review_content = future.result()
                total_cost += new_total_cost  # Accumulate cost
                # Accumulate review content
                collected_reviews.append(review_content)

            except Exception as e:
                logging.error(f"Error during OpenAI API call: {e}")
                continue  # Proceed with the next chunk if an exception occurs

    with open(output_path, 'a', encoding='utf-8') as md_file:
        for review_content in collected_reviews:
            md_file.write(review_content)

    logging.info(f"Review written to {output_path}")
    return total_cost


def main():
    """
    Main function to set up the tool, parse arguments, and start the review process.

    It initializes the review process by setting up necessary parameters, scanning the repository, and invoking the review process for each identified file.
    """
    parser = argparse.ArgumentParser(description="Code Review Tool")
    parser.add_argument("repo_path", type=str, help="Path to the code repository")
    parser.add_argument("--model_name", type=str, default="gpt-4-0125-preview", help="OpenAI model name")
    parser.add_argument("--language", type=str, default="CPP", help="Programming language (e.g. cpp, python, js)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for Markdown output")
    parser.add_argument("--chunk_size", type=int, default=0, help="Chunk size for text splitter")
    parser.add_argument("--estimate_cost", type=bool, default=False, help="Run check only to see maybe cost")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)

    # Assuming LANGUAGE_FILE_EXTENSIONS and other necessary setups are defined
    file_extensions = LANGUAGE_FILE_EXTENSIONS.get(args.language.upper(), [])
    if not file_extensions:
        logging.error(f"Unsupported language: {args.language}. Supported languages are: {', '.join(LANGUAGE_FILE_EXTENSIONS.keys())}")
        return

    chunk_size = args.chunk_size
    if args.chunk_size == 0:
        chunk_size = get_chunk_size(args.model_name)

    files_to_review = process_repository(args.repo_path, file_extensions)

    total_cost = 0
    global ONLY_CHECK_COST
    if args.estimate_cost:
        ONLY_CHECK_COST = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(review_code_with_openai, file_path, args.model_name, args.output_dir, args.language, chunk_size)
                   for file_path in files_to_review]
        for future in concurrent.futures.as_completed(futures):
            total_cost += future.result()
            logging.info(f"Current cost: ${total_cost:.4f}")

    logging.info(f"Review process completed. Total cost: ${total_cost:.4f}.")

if __name__ == "__main__":
    main()
