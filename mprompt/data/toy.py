from typing import List

def generate_synthetic_data() -> List[str]:
    s = []
    for k in TASKS_TOY.keys():
        s.append(' '.join(TASKS_TOY[k]['examples']))
    return s


TASKS_TOY = {
    # Observations: yes/no questions are terrible
    # 'template': 'Does the input contain an animal?\nInput: {input}\nAnswer (yes or no):',
    # 'target_token': ' yes',

    'toy_animal': {
        'check_func': r'animal',
        'groundtruth_explanation': 'Return whether the input is an animal.',
        'template': 'A {input} is a type of',
        'target_token': ' animal',
        'get_relevant_data': generate_synthetic_data,
        'examples': ['cat', 'dog', 'giraffe', 'horse', 'zebra', 'raccoon'],
    },
    'toy_food': {
        'check_func': r'fruit|edible',
        'groundtruth_explanation': 'Return whether the input is a food.',
        'template': '{input} is a type of',
        'target_token': ' food',
        'get_relevant_data': generate_synthetic_data,
        'examples': ['apple', 'orange', 'pear', 'pizza', 'lasagna', 'curry', 'salad', 'chopstick'],
    },
    'toy_numbers': {
        'check_func': r'number',
        'groundtruth_explanation': 'Return whether the input is a number.',
        'template': '{input} is related to the concept of',
        'target_token': ' numbers',
        'get_relevant_data': generate_synthetic_data,
        # actual numbers like '1' do poorly
        'examples': ['1', '2', '3', 'four', 'five', 'six', 'plus', 'minus', 'divide'],
    }
}