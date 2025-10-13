from math_verify import parse, verify


def verify_answer(result, answer):
    parsed_result = parse(result)
    parsed_answer = parse(answer)
    correct = verify(parsed_answer, parsed_result)
    return {'correct': correct}


def math_reward_function(response, target):
    correct = verify_answer(response, target)
    return 1 if correct['correct'] else 0