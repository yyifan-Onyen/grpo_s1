from datasets import load_dataset
import io
import os
import re
import signal
import tempfile
import contextlib



def create_mbpp_dataset():
    def process_example(example):
        instruction = 'You are a helpful assistant solving coding problems for users. You should provide a complete Python function to meet the requirements. Your code should be enclosed with a Markdown code block. For example, ```python\ncode here\n```.'
        checker = f'{example["test_setup_code"]}\n' + '\n'.join(example['test_list'])
        problem = f'{example["text"]}\nYour code should pass the following unit tests:\n```python\n{checker}\n```'
        solution = f'```python\n{example["code"]}\n```'
        return {'instruction': instruction, 'problem': problem,
                'solution': solution, 'checker': checker}
    dataset = load_dataset('Muennighoff/mbpp', name='full', split='test')
    dataset = dataset.map(process_example)
    dataset = dataset.remove_columns(['task_id', 'text', 'code', 'test_list',
                                      'test_setup_code', 'challenge_test_list'])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return dataset


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = 'stdin'


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException('Timed out!')
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root):
    if root == '.':
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def unsafe_execute(program: str, timeout: float = 5.0):
    with create_tempdir():
        try:
            globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(program, globals)
            status = 'passed'
        except TimeoutException:
            status = 'failed: time out'
        except BaseException as e:
            status = f'failed: {e}'
    return status


def verify_code(code: str, checker: str, timeout: float = 5.0):
    program = f'from typing import *\n{code}\n{checker}'
    status = unsafe_execute(program, timeout)
    if status == 'passed':
        correct = True
    else:
        correct = False
    return {'correct': correct, 'status': status}


def verify_answer(answer: str, checker: str, timeout: float = 5.0):
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = pattern.findall(answer)
    if matches:
        code = matches[-1]
        formatted = True
        result = verify_code(code, checker, timeout)
        correct = result['correct']
        status = result['status']
    else:
        formatted = False
        correct = False
        status = 'failed: invalid format'
    return {'formatted': formatted, 'correct': correct, 'status': status}


if __name__ == '__main__':
    dataset = create_mbpp_dataset()
    print(dataset['train'][0])