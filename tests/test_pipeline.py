import os
from os.path import dirname, join
repo_dir = dirname(dirname(os.path.abspath(__file__)))
save_dir = os.path.join(repo_dir, 'results', 'tmp')
prefix = f'PYTHONPATH={join(repo_dir, "experiments")}'


def test_small_pipeline():
    cmd = prefix + ' python ' + \
        os.path.join(repo_dir, 'experiments',
                     f'01_explain.py --use_cache 0 --save_dir {save_dir}')
    print(cmd)
    exit_value = os.system(cmd)
    assert exit_value == 0, 'default pipeline passed'


def test_gradient_pipeline():
    cmd = prefix + ' python ' + \
        os.path.join(repo_dir, 'experiments',
                     f'01_explain.py --use_cache 0 --save_dir {save_dir} --method_name gradient')
    print(cmd)
    exit_value = os.system(cmd)
    assert exit_value == 0, 'default pipeline passed'


if __name__ == '__main__':
    test_gradient_pipeline()
