from datasets import load_dataset
from math_verify import parse, verify
import re
"""
This script is used to create a dataset of GSM8K problems and solutions.
It also contains a function to verify the answer of a problem.
"""


def create_gsm8k_dataset():
    def process_example(example):
        instruction = f'You are a helpful assistant solving mathematical problems for users. You should provide a complete and correct solution. The final answer should be boxed in LaTeX format. For example, "The final answer is $\\boxed{{answer here}}$".'
        problem = f'{example["question"]}'
        solution = f'{example["answer"]}'
        prompt = instruction + "\n\n" + problem
        return {'prompt': prompt, 'solution': solution}
    dataset = load_dataset("openai/gsm8k", name='main')
    dataset = dataset.map(process_example)
    dataset = dataset.remove_columns(['question', 'answer'])
    return dataset


def verify_answer(result, answer):
    parsed_result = parse(result)
    parsed_answer = parse(answer)
    correct = verify(parsed_answer, parsed_result)
    return {'correct': correct}


def gsm8k_reward_function(response, target, end_token):
    if end_token in response:
        response = response.split(end_token)[0]
    correct = verify_answer(response, target)
    return 1 if correct['correct'] else 0


def main():
    dataset = create_gsm8k_dataset()
    print(dataset['train'][0])
    #测试一下gsm8k reward function
    response = "The final answer is $\\boxed{10}$"
    target = "10"
    end_token = "$\\\\boxed{}$"
    print(gsm8k_reward_function(response, target, end_token))

if __name__ == "__main__":
    main()