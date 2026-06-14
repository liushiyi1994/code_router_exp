# RouteCode Code Cards

These cards summarize route labels learned from train-set utility profiles. They are synthetic-pilot diagnostics, not paper claims.

## Route label 0: `broad_knowledge__Intern-S1-mini`

- Size: 46 train queries
- Best model: `Intern-S1-mini`
- Second-best model: `MiniCPM4.1-8B`
- Mean utility margin: 0.6522
- Dominant domains: broad_knowledge (18), science (10), code (10)
- Dominant datasets: mmlupro (18), gpqa (10), mbpp (10)
- Model utility vector: Intern-S1-mini=1.000, MiniCPM4.1-8B=0.348, Qwen2.5-Coder-7B-Instruct=0.152, DeepSeek-R1-Distill-Qwen-7B=0.043, Llama-3.1-8B-Instruct=0.022, Qwen3-8B=0.000
- Human-readable explanation: `broad_knowledge__Intern-S1-mini` groups queries whose train-set utility profile favors `Intern-S1-mini`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - Assume all gases are perfect unless stated otherwise. Note that 1 atm = 1.013 25 bar. Unless otherwise stated, thermochemical data are for 298.15 K. Concerns over the harmful effects of chlorofluorocarbons on stratospheric ozone have motivated a search for new refrigerants. One such alternative is 2,2-dichloro-1,1,1-trifluoroethane (refrigerant 123). Younglove and McLinden published a compendium of thermophysical properties of this substance (J. Phys. Chem. Ref. Data 23, 7 (1994)), from which properties such as the Joule-Thomson coefficient $\mu$ can be computed. Compute the temperature change that would accompany adiabatic expansion of $2.0 \mathrm{~mol}$ of this refrigerant from $1.5 \mathrm{bar}$ to 0.5 bar at $50^{\circ} \mathrm{C}$.
  - The model of light most supported by the photoelectric effect is the
  - Which of the following are advantages of the VAR approach to modelling the relationship between variables relative to the estimation of full structural models?

i) VARs receive strong motivation from financial and economic theory


ii) VARs in their reduced forms can be used easily to produce time-series forecasts


iii) VAR models are typically highly parsimonious


iv) OLS can be applied separately to each equation in a reduced form VAR
  - a) The maximum pressure variation P that the ear can tolerate in loud sounds is about 28 N / m^2 (=28 Pa). Normal atmospheric pressure is about 100,000 Pa. Find the corresponding maximum displacement for a sound wave in air having a frequency of 1000 Hz. b) In the faintest sound that can be heard at 1000 Hz the pressure amplitude is about 2.0 × 10^-5 Pa. Find the correspond-ing displacement amplitude.
- Highest-regret train examples under this label:
  - Assume all gases are perfect unless stated otherwise. Note that 1 atm = 1.013 25 bar. Unless otherwise stated, thermochemical data are for 298.15 K. Concerns over the harmful effects of chlorofluorocarbons on stratospheric ozone have motivated a search for new refrigerants. One such alternative is 2,2-dichloro-1,1,1-trifluoroethane (refrigerant 123). Younglove and McLinden published a compendium of thermophysical properties of this substance (J. Phys. Chem. Ref. Data 23, 7 (1994)), from which properties such as the Joule-Thomson coefficient $\mu$ can be computed. Compute the temperature change that would accompany adiabatic expansion of $2.0 \mathrm{~mol}$ of this refrigerant from $1.5 \mathrm{bar}$ to 0.5 bar at $50^{\circ} \mathrm{C}$.
  - The model of light most supported by the photoelectric effect is the
  - Which of the following are advantages of the VAR approach to modelling the relationship between variables relative to the estimation of full structural models?

i) VARs receive strong motivation from financial and economic theory


ii) VARs in their reduced forms can be used easily to produce time-series forecasts


iii) VAR models are typically highly parsimonious


iv) OLS can be applied separately to each equation in a reduced form VAR
  - a) The maximum pressure variation P that the ear can tolerate in loud sounds is about 28 N / m^2 (=28 Pa). Normal atmospheric pressure is about 100,000 Pa. Find the corresponding maximum displacement for a sound wave in air having a frequency of 1000 Hz. b) In the faintest sound that can be heard at 1000 Hz the pressure amplitude is about 2.0 × 10^-5 Pa. Find the correspond-ing displacement amplitude.

## Route label 1: `math__Intern-S1-mini`

- Size: 328 train queries
- Best model: `Intern-S1-mini`
- Second-best model: `MiniCPM4.1-8B`
- Mean utility margin: 0.0000
- Dominant domains: math (122), broad_knowledge (108), code (95)
- Dominant datasets: math500 (122), mmlupro (108), mbpp (80)
- Model utility vector: Intern-S1-mini=1.000, MiniCPM4.1-8B=1.000, Llama-3.1-8B-Instruct=1.000, Qwen2.5-Coder-7B-Instruct=1.000, DeepSeek-R1-Distill-Qwen-7B=1.000, Qwen3-8B=0.942
- Human-readable explanation: `math__Intern-S1-mini` groups queries whose train-set utility profile favors `Intern-S1-mini`. It is most associated with domain `math` and dataset `math500` in this run.
- Representative queries:
  - from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """

  - from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """

  - from typing import List, Tuple


def rolling_max(numbers: List[int]) -> List[int]:
    """ From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """

  - from typing import List


def all_prefixes(string: str) -> List[str]:
    """ Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes('abc')
    ['a', 'ab', 'abc']
    """

- Highest-regret train examples under this label:
  - from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """

  - from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """

  - from typing import List, Tuple


def rolling_max(numbers: List[int]) -> List[int]:
    """ From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """

  - from typing import List


def all_prefixes(string: str) -> List[str]:
    """ Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes('abc')
    ['a', 'ab', 'abc']
    """


## Route label 2: `code__Qwen2.5-Coder-7B-Instruct`

- Size: 78 train queries
- Best model: `Qwen2.5-Coder-7B-Instruct`
- Second-best model: `DeepSeek-R1-Distill-Qwen-7B`
- Mean utility margin: 0.9103
- Dominant domains: code (62), broad_knowledge (9), science (7)
- Dominant datasets: mbpp (49), humaneval (13), mmlupro (9)
- Model utility vector: Qwen2.5-Coder-7B-Instruct=1.000, DeepSeek-R1-Distill-Qwen-7B=0.090, MiniCPM4.1-8B=0.051, Qwen3-8B=0.000, Intern-S1-mini=0.000, Llama-3.1-8B-Instruct=0.000
- Human-readable explanation: `code__Qwen2.5-Coder-7B-Instruct` groups queries whose train-set utility profile favors `Qwen2.5-Coder-7B-Instruct`. It is most associated with domain `code` and dataset `mbpp` in this run.
- Representative queries:
  - 

def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """

  - 
def strange_sort_list(lst):
    '''
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Examples:
    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
    strange_sort_list([]) == []
    '''

  - 
def iscube(a):
    '''
    Write a function that takes an integer a and returns True 
    if this ingeger is a cube of some integer number.
    Note: you may assume the input is always valid.
    Examples:
    iscube(1) ==> True
    iscube(2) ==> False
    iscube(-1) ==> True
    iscube(64) ==> True
    iscube(0) ==> True
    iscube(180) ==> False
    '''

  - 
def numerical_letter_grade(grades):
    """It is the last week of the semester and the teacher has to give the grades
    to students. The teacher has been making her own algorithm for grading.
    The only problem is, she has lost the code she used for grading.
    She has given you a list of GPAs for some students and you have to write 
    a function that can output a list of letter grades using the following table:
             GPA       |    Letter grade
              4.0                A+
            > 3.7                A 
            > 3.3                A- 
            > 3.0                B+
            > 2.7                B 
            > 2.3                B-
            > 2.0                C+
            > 1.7                C
            > 1.3                C-
            > 1.0                D+ 
            > 0.7                D 
            > 0.0                D-
              0.0                E
    

    Example:
    grade_equation([4.0, 3, 1.7, 2, 3.5]) ==> ['A+', 'B', 'C-', 'C', 'A-']
    """

- Highest-regret train examples under this label:
  - 

def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """

  - 
def strange_sort_list(lst):
    '''
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Examples:
    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
    strange_sort_list([]) == []
    '''

  - 
def iscube(a):
    '''
    Write a function that takes an integer a and returns True 
    if this ingeger is a cube of some integer number.
    Note: you may assume the input is always valid.
    Examples:
    iscube(1) ==> True
    iscube(2) ==> False
    iscube(-1) ==> True
    iscube(64) ==> True
    iscube(0) ==> True
    iscube(180) ==> False
    '''

  - 
def numerical_letter_grade(grades):
    """It is the last week of the semester and the teacher has to give the grades
    to students. The teacher has been making her own algorithm for grading.
    The only problem is, she has lost the code she used for grading.
    She has given you a list of GPAs for some students and you have to write 
    a function that can output a list of letter grades using the following table:
             GPA       |    Letter grade
              4.0                A+
            > 3.7                A 
            > 3.3                A- 
            > 3.0                B+
            > 2.7                B 
            > 2.3                B-
            > 2.0                C+
            > 1.7                C
            > 1.3                C-
            > 1.0                D+ 
            > 0.7                D 
            > 0.0                D-
              0.0                E
    

    Example:
    grade_equation([4.0, 3, 1.7, 2, 3.5]) ==> ['A+', 'B', 'C-', 'C', 'A-']
    """


## Route label 3: `math__DeepSeek-R1-Distill-Qwen-7B`

- Size: 150 train queries
- Best model: `DeepSeek-R1-Distill-Qwen-7B`
- Second-best model: `Qwen3-8B`
- Mean utility margin: 0.0867
- Dominant domains: math (72), broad_knowledge (54), science (14)
- Dominant datasets: math500 (61), mmlupro (54), gpqa (14)
- Model utility vector: DeepSeek-R1-Distill-Qwen-7B=1.000, Qwen3-8B=0.913, MiniCPM4.1-8B=0.913, Intern-S1-mini=0.793, Qwen2.5-Coder-7B-Instruct=0.000, Llama-3.1-8B-Instruct=0.000
- Human-readable explanation: `math__DeepSeek-R1-Distill-Qwen-7B` groups queries whose train-set utility profile favors `DeepSeek-R1-Distill-Qwen-7B`. It is most associated with domain `math` and dataset `math500` in this run.
- Representative queries:
  - 
def max_fill(grid, capacity):
    import math
    """
    You are given a rectangular grid of wells. Each row represents a single well,
    and each 1 in a row represents a single unit of water.
    Each well has a corresponding bucket that can be used to extract water from it, 
    and all buckets have the same capacity.
    Your task is to use the buckets to empty the wells.
    Output the number of times you need to lower the buckets.

    Example 1:
        Input: 
            grid : [[0,0,1,0], [0,1,0,0], [1,1,1,1]]
            bucket_capacity : 1
        Output: 6

    Example 2:
        Input: 
            grid : [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]
            bucket_capacity : 2
        Output: 5
    
    Example 3:
        Input: 
            grid : [[0,0,0], [0,0,0]]
            bucket_capacity : 5
        Output: 0

    Constraints:
        * all wells have the same length
        * 1 <= grid.length <= 10^2
        * 1 <= grid[:,1].length <= 10^2
        * grid[i][j] -> 0 | 1
        * 1 <= capacity <= 10
    """

  - 
def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers
    Example
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """

  - A noninterest-bearing note with a face value of $600 and a term of 30 days dated April 5 was discounted April 15 at a rate of 5%. What were the proceeds?
  - Star Co. is a retail store specializing in contemporary furniture. The following information is taken from Star's June budget: Sales $540000 Cost of goods sold 300000 Merchandise inventory‚ÄìJune 1 150000 Merchandise inventory‚ÄìJune 30 180000 Accounts payable for purchases‚ÄìJune 1 85000 Accounts payable for purchases‚ÄìJune 30 75000 What amount should Star budget for cash disbursements for June purchases?
- Highest-regret train examples under this label:
  - 
def max_fill(grid, capacity):
    import math
    """
    You are given a rectangular grid of wells. Each row represents a single well,
    and each 1 in a row represents a single unit of water.
    Each well has a corresponding bucket that can be used to extract water from it, 
    and all buckets have the same capacity.
    Your task is to use the buckets to empty the wells.
    Output the number of times you need to lower the buckets.

    Example 1:
        Input: 
            grid : [[0,0,1,0], [0,1,0,0], [1,1,1,1]]
            bucket_capacity : 1
        Output: 6

    Example 2:
        Input: 
            grid : [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]
            bucket_capacity : 2
        Output: 5
    
    Example 3:
        Input: 
            grid : [[0,0,0], [0,0,0]]
            bucket_capacity : 5
        Output: 0

    Constraints:
        * all wells have the same length
        * 1 <= grid.length <= 10^2
        * 1 <= grid[:,1].length <= 10^2
        * grid[i][j] -> 0 | 1
        * 1 <= capacity <= 10
    """

  - 
def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers
    Example
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """

  - A noninterest-bearing note with a face value of $600 and a term of 30 days dated April 5 was discounted April 15 at a rate of 5%. What were the proceeds?
  - Star Co. is a retail store specializing in contemporary furniture. The following information is taken from Star's June budget: Sales $540000 Cost of goods sold 300000 Merchandise inventory‚ÄìJune 1 150000 Merchandise inventory‚ÄìJune 30 180000 Accounts payable for purchases‚ÄìJune 1 85000 Accounts payable for purchases‚ÄìJune 30 75000 What amount should Star budget for cash disbursements for June purchases?

## Route label 4: `code__Qwen3-8B`

- Size: 61 train queries
- Best model: `Qwen3-8B`
- Second-best model: `Qwen2.5-Coder-7B-Instruct`
- Mean utility margin: 0.0000
- Dominant domains: code (40), broad_knowledge (15), science (5)
- Dominant datasets: mbpp (32), mmlupro (15), humaneval (8)
- Model utility vector: Qwen3-8B=1.000, Qwen2.5-Coder-7B-Instruct=1.000, MiniCPM4.1-8B=0.246, Intern-S1-mini=0.180, DeepSeek-R1-Distill-Qwen-7B=0.180, Llama-3.1-8B-Instruct=0.000
- Human-readable explanation: `code__Qwen3-8B` groups queries whose train-set utility profile favors `Qwen3-8B`. It is most associated with domain `code` and dataset `mbpp` in this run.
- Representative queries:
  - 

def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """

  - 

def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are three distinct elements in the list that
    sum to zero, and False otherwise.

    >>> triples_sum_to_zero([1, 3, 5, 0])
    False
    >>> triples_sum_to_zero([1, 3, -2, 1])
    True
    >>> triples_sum_to_zero([1, 2, 3, 7])
    False
    >>> triples_sum_to_zero([2, 4, -5, 3, 9, 7])
    True
    >>> triples_sum_to_zero([1])
    False
    """

  - 
def get_row(lst, x):
    """
    You are given a 2 dimensional data, as a nested lists,
    which is similar to matrix, however, unlike matrices,
    each row may contain a different number of columns.
    Given lst, and integer x, find integers x in the list,
    and return list of tuples, [(x1, y1), (x2, y2) ...] such that
    each tuple is a coordinate - (row, columns), starting with 0.
    Sort coordinates initially by rows in ascending order.
    Also, sort coordinates of the row by columns in descending order.
    
    Examples:
    get_row([
      [1,2,3,4,5,6],
      [1,2,3,4,1,6],
      [1,2,3,4,5,1]
    ], 1) == [(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]
    get_row([], 1) == []
    get_row([[], [1], [1, 2, 3]], 3) == [(2, 2)]
    """

  - 
def rounded_avg(n, m):
    """You are given two positive integers n and m, and your task is to compute the
    average of the integers from n through m (including n and m). 
    Round the answer to the nearest integer and convert that to binary.
    If n is greater than m, return -1.
    Example:
    rounded_avg(1, 5) => "0b11"
    rounded_avg(7, 5) => -1
    rounded_avg(10, 20) => "0b1111"
    rounded_avg(20, 33) => "0b11010"
    """

- Highest-regret train examples under this label:
  - 

def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """

  - 

def triples_sum_to_zero(l: list):
    """
    triples_sum_to_zero takes a list of integers as an input.
    it returns True if there are three distinct elements in the list that
    sum to zero, and False otherwise.

    >>> triples_sum_to_zero([1, 3, 5, 0])
    False
    >>> triples_sum_to_zero([1, 3, -2, 1])
    True
    >>> triples_sum_to_zero([1, 2, 3, 7])
    False
    >>> triples_sum_to_zero([2, 4, -5, 3, 9, 7])
    True
    >>> triples_sum_to_zero([1])
    False
    """

  - 
def get_row(lst, x):
    """
    You are given a 2 dimensional data, as a nested lists,
    which is similar to matrix, however, unlike matrices,
    each row may contain a different number of columns.
    Given lst, and integer x, find integers x in the list,
    and return list of tuples, [(x1, y1), (x2, y2) ...] such that
    each tuple is a coordinate - (row, columns), starting with 0.
    Sort coordinates initially by rows in ascending order.
    Also, sort coordinates of the row by columns in descending order.
    
    Examples:
    get_row([
      [1,2,3,4,5,6],
      [1,2,3,4,1,6],
      [1,2,3,4,5,1]
    ], 1) == [(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]
    get_row([], 1) == []
    get_row([[], [1], [1, 2, 3]], 3) == [(2, 2)]
    """

  - 
def rounded_avg(n, m):
    """You are given two positive integers n and m, and your task is to compute the
    average of the integers from n through m (including n and m). 
    Round the answer to the nearest integer and convert that to binary.
    If n is greater than m, return -1.
    Example:
    rounded_avg(1, 5) => "0b11"
    rounded_avg(7, 5) => -1
    rounded_avg(10, 20) => "0b1111"
    rounded_avg(20, 33) => "0b11010"
    """


## Route label 5: `code__Llama-3.1-8B-Instruct`

- Size: 65 train queries
- Best model: `Llama-3.1-8B-Instruct`
- Second-best model: `Qwen2.5-Coder-7B-Instruct`
- Mean utility margin: 0.0000
- Dominant domains: code (54), broad_knowledge (7), math (2)
- Dominant datasets: mbpp (41), humaneval (13), mmlupro (7)
- Model utility vector: Llama-3.1-8B-Instruct=1.000, Qwen2.5-Coder-7B-Instruct=1.000, DeepSeek-R1-Distill-Qwen-7B=0.185, Intern-S1-mini=0.108, MiniCPM4.1-8B=0.062, Qwen3-8B=0.000
- Human-readable explanation: `code__Llama-3.1-8B-Instruct` groups queries whose train-set utility profile favors `Llama-3.1-8B-Instruct`. It is most associated with domain `code` and dataset `mbpp` in this run.
- Representative queries:
  - from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """

  - from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """

  - from typing import List


def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """

  - from typing import List, Any


def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """

- Highest-regret train examples under this label:
  - from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """

  - from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """

  - from typing import List


def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """

  - from typing import List, Any


def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """


## Route label 6: `broad_knowledge__MiniCPM4.1-8B`

- Size: 232 train queries
- Best model: `MiniCPM4.1-8B`
- Second-best model: `DeepSeek-R1-Distill-Qwen-7B`
- Mean utility margin: 0.0776
- Dominant domains: broad_knowledge (106), code (89), science (24)
- Dominant datasets: mmlupro (106), mbpp (77), gpqa (24)
- Model utility vector: MiniCPM4.1-8B=0.103, DeepSeek-R1-Distill-Qwen-7B=0.026, Intern-S1-mini=0.000, Qwen3-8B=0.000, Qwen2.5-Coder-7B-Instruct=0.000, Llama-3.1-8B-Instruct=0.000
- Human-readable explanation: `broad_knowledge__MiniCPM4.1-8B` groups queries whose train-set utility profile favors `MiniCPM4.1-8B`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - 

def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """

  - 

def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """

  - 
def is_multiply_prime(a):
    """Write a function that returns true if the given number is the multiplication of 3 prime numbers
    and false otherwise.
    Knowing that (a) is less then 100. 
    Example:
    is_multiply_prime(30) == True
    30 = 2 * 3 * 5
    """

  - 
def is_simple_power(x, n):
    """Your task is to write a function that returns true if a number x is a simple
    power of n and false in other cases.
    x is a simple power of n if n**int=x
    For example:
    is_simple_power(1, 4) => true
    is_simple_power(2, 2) => true
    is_simple_power(8, 2) => true
    is_simple_power(3, 2) => false
    is_simple_power(3, 1) => false
    is_simple_power(5, 3) => false
    """

- Highest-regret train examples under this label:
  - 

def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """

  - A customer at a fish market was leaving the store after purchasing an assortment of shrimp, oysters, and scallops. He was walking along the sidewalk in front of the store when he slipped on a piece of eel. He brought suit against the owner of the market claiming that he suffered leg and back injuries. The owner, although admitting that the customer was injured by slipping on the eel, denied negligence and claimed that the customer was contributorily negligent. At trial, the owner calls a witness to testify that before the fall he heard someone call out to the customer, "Watch it, buddy, you're going to step on that piece of fish. "The witness's testimony is
  - Which of the following cases best illustrates the 'living instrument principle' used by the European Court of Human Rights?
  - The "c" in the word cat is best described as a

## Route label 7: `code__Qwen3-8B`

- Size: 88 train queries
- Best model: `Qwen3-8B`
- Second-best model: `Llama-3.1-8B-Instruct`
- Mean utility margin: 0.0000
- Dominant domains: code (68), broad_knowledge (14), science (5)
- Dominant datasets: mbpp (51), humaneval (17), mmlupro (14)
- Model utility vector: Qwen3-8B=1.000, Llama-3.1-8B-Instruct=1.000, Qwen2.5-Coder-7B-Instruct=0.761, Intern-S1-mini=0.432, MiniCPM4.1-8B=0.148, DeepSeek-R1-Distill-Qwen-7B=0.000
- Human-readable explanation: `code__Qwen3-8B` groups queries whose train-set utility profile favors `Qwen3-8B`. It is most associated with domain `code` and dataset `mbpp` in this run.
- Representative queries:
  - from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """

  - 

def median(l: list):
    """Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """

  - 

def monotonic(l: list):
    """Return True is list elements are monotonically increasing or decreasing.
    >>> monotonic([1, 2, 4, 20])
    True
    >>> monotonic([1, 20, 4, 10])
    False
    >>> monotonic([4, 1, 0, -10])
    True
    """

  - 

def fibfib(n: int):
    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    Please write a function to efficiently compute the n-th element of the fibfib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24
    """

- Highest-regret train examples under this label:
  - from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """

  - 

def median(l: list):
    """Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """

  - 

def monotonic(l: list):
    """Return True is list elements are monotonically increasing or decreasing.
    >>> monotonic([1, 2, 4, 20])
    True
    >>> monotonic([1, 20, 4, 10])
    False
    >>> monotonic([4, 1, 0, -10])
    True
    """

  - 

def fibfib(n: int):
    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    Please write a function to efficiently compute the n-th element of the fibfib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24
    """


## Route label 8: `math__DeepSeek-R1-Distill-Qwen-7B`

- Size: 163 train queries
- Best model: `DeepSeek-R1-Distill-Qwen-7B`
- Second-best model: `Qwen2.5-Coder-7B-Instruct`
- Mean utility margin: 0.0000
- Dominant domains: math (63), broad_knowledge (50), code (42)
- Dominant datasets: math500 (63), mmlupro (50), mbpp (37)
- Model utility vector: DeepSeek-R1-Distill-Qwen-7B=1.000, Qwen2.5-Coder-7B-Instruct=1.000, Qwen3-8B=0.939, MiniCPM4.1-8B=0.877, Intern-S1-mini=0.865, Llama-3.1-8B-Instruct=0.000
- Human-readable explanation: `math__DeepSeek-R1-Distill-Qwen-7B` groups queries whose train-set utility profile favors `DeepSeek-R1-Distill-Qwen-7B`. It is most associated with domain `math` and dataset `math500` in this run.
- Representative queries:
  - from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """

  - 

def remove_vowels(text):
    """
    remove_vowels is a function that takes string and returns string without vowels.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\nghijklm")
    'bcdf\nghjklm'
    >>> remove_vowels('abcdef')
    'bcdf'
    >>> remove_vowels('aaaaa')
    ''
    >>> remove_vowels('aaBAA')
    'B'
    >>> remove_vowels('zbcd')
    'zbcd'
    """

  - 
def make_a_pile(n):
    """
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list, where element at index
    i represents the number of stones in the level (i+1).

    Examples:
    >>> make_a_pile(3)
    [3, 5, 7]
    """

  - 
def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.

    Example 1:

        Input: arr = [-3, -4, 5], k = 3
        Output: [-4, -3, 5]

    Example 2:

        Input: arr = [4, -4, 4], k = 2
        Output: [4, 4]

    Example 3:

        Input: arr = [-3, 2, 1, 2, -1, -2, 1], k = 1
        Output: [2]

    Note:
        1. The length of the array will be in the range of [1, 1000].
        2. The elements in the array will be in the range of [-1000, 1000].
        3. 0 <= k <= len(arr)
    """

- Highest-regret train examples under this label:
  - from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """

  - 

def remove_vowels(text):
    """
    remove_vowels is a function that takes string and returns string without vowels.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\nghijklm")
    'bcdf\nghjklm'
    >>> remove_vowels('abcdef')
    'bcdf'
    >>> remove_vowels('aaaaa')
    ''
    >>> remove_vowels('aaBAA')
    'B'
    >>> remove_vowels('zbcd')
    'zbcd'
    """

  - 
def make_a_pile(n):
    """
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list, where element at index
    i represents the number of stones in the level (i+1).

    Examples:
    >>> make_a_pile(3)
    [3, 5, 7]
    """

  - 
def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.

    Example 1:

        Input: arr = [-3, -4, 5], k = 3
        Output: [-4, -3, 5]

    Example 2:

        Input: arr = [4, -4, 4], k = 2
        Output: [4, 4]

    Example 3:

        Input: arr = [-3, 2, 1, 2, -1, -2, 1], k = 1
        Output: [2]

    Note:
        1. The length of the array will be in the range of [1, 1000].
        2. The elements in the array will be in the range of [-1000, 1000].
        3. 0 <= k <= len(arr)
    """


## Route label 9: `code__Llama-3.1-8B-Instruct`

- Size: 115 train queries
- Best model: `Llama-3.1-8B-Instruct`
- Second-best model: `DeepSeek-R1-Distill-Qwen-7B`
- Mean utility margin: 0.0000
- Dominant domains: code (111), science (3), broad_knowledge (1)
- Dominant datasets: mbpp (101), humaneval (10), gpqa (3)
- Model utility vector: Llama-3.1-8B-Instruct=1.000, DeepSeek-R1-Distill-Qwen-7B=1.000, Qwen2.5-Coder-7B-Instruct=0.922, Qwen3-8B=0.887, Intern-S1-mini=0.791, MiniCPM4.1-8B=0.000
- Human-readable explanation: `code__Llama-3.1-8B-Instruct` groups queries whose train-set utility profile favors `Llama-3.1-8B-Instruct`. It is most associated with domain `code` and dataset `mbpp` in this run.
- Representative queries:
  - from typing import List, Optional


def longest(strings: List[str]) -> Optional[str]:
    """ Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """

  - from typing import List


def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """

  - from typing import List


def factorize(n: int) -> List[int]:
    """ Return list of prime factors of given integer in the order from smallest to largest.
    Each of the factors should be listed number of times corresponding to how many times it appeares in factorization.
    Input number should be equal to the product of all factors
    >>> factorize(8)
    [2, 2, 2]
    >>> factorize(25)
    [5, 5]
    >>> factorize(70)
    [2, 5, 7]
    """

  - 

def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """

- Highest-regret train examples under this label:
  - from typing import List, Optional


def longest(strings: List[str]) -> Optional[str]:
    """ Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """

  - from typing import List


def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """

  - from typing import List


def factorize(n: int) -> List[int]:
    """ Return list of prime factors of given integer in the order from smallest to largest.
    Each of the factors should be listed number of times corresponding to how many times it appeares in factorization.
    Input number should be equal to the product of all factors
    >>> factorize(8)
    [2, 2, 2]
    >>> factorize(25)
    [5, 5]
    >>> factorize(70)
    [2, 5, 7]
    """

  - 

def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """


## Route label 10: `code__Qwen2.5-Coder-7B-Instruct`

- Size: 76 train queries
- Best model: `Qwen2.5-Coder-7B-Instruct`
- Second-best model: `MiniCPM4.1-8B`
- Mean utility margin: 0.0000
- Dominant domains: code (52), broad_knowledge (20), math (3)
- Dominant datasets: mbpp (46), mmlupro (20), humaneval (6)
- Model utility vector: Qwen2.5-Coder-7B-Instruct=1.000, MiniCPM4.1-8B=1.000, DeepSeek-R1-Distill-Qwen-7B=1.000, Llama-3.1-8B-Instruct=0.934, Qwen3-8B=0.513, Intern-S1-mini=0.000
- Human-readable explanation: `code__Qwen2.5-Coder-7B-Instruct` groups queries whose train-set utility profile favors `Qwen2.5-Coder-7B-Instruct`. It is most associated with domain `code` and dataset `mbpp` in this run.
- Representative queries:
  - 

def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """

  - 

def car_race_collision(n: int):
    """
    Imagine a road that's a perfectly straight infinitely long line.
    n cars are driving left to right;  simultaneously, a different set of n cars
    are driving right to left.   The two sets of cars start out being very far from
    each other.  All cars move in the same speed.  Two cars are said to collide
    when a car that's moving left to right hits a car that's moving right to left.
    However, the cars are infinitely sturdy and strong; as a result, they continue moving
    in their trajectory as if they did not collide.

    This function outputs the number of such collisions.
    """

  - 

def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("<")
    False
    >>> correct_bracketing("<>")
    True
    >>> correct_bracketing("<<><>>")
    True
    >>> correct_bracketing("><<>")
    False
    """

  - 

def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """

- Highest-regret train examples under this label:
  - 

def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """

  - 

def car_race_collision(n: int):
    """
    Imagine a road that's a perfectly straight infinitely long line.
    n cars are driving left to right;  simultaneously, a different set of n cars
    are driving right to left.   The two sets of cars start out being very far from
    each other.  All cars move in the same speed.  Two cars are said to collide
    when a car that's moving left to right hits a car that's moving right to left.
    However, the cars are infinitely sturdy and strong; as a result, they continue moving
    in their trajectory as if they did not collide.

    This function outputs the number of such collisions.
    """

  - 

def correct_bracketing(brackets: str):
    """ brackets is a string of "<" and ">".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("<")
    False
    >>> correct_bracketing("<>")
    True
    >>> correct_bracketing("<<><>>")
    True
    >>> correct_bracketing("><<>")
    False
    """

  - 

def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """


## Route label 11: `broad_knowledge__Llama-3.1-8B-Instruct`

- Size: 67 train queries
- Best model: `Llama-3.1-8B-Instruct`
- Second-best model: `MiniCPM4.1-8B`
- Mean utility margin: 0.0000
- Dominant domains: broad_knowledge (30), math (25), science (7)
- Dominant datasets: mmlupro (30), math500 (21), gpqa (7)
- Model utility vector: Llama-3.1-8B-Instruct=1.000, MiniCPM4.1-8B=1.000, DeepSeek-R1-Distill-Qwen-7B=1.000, Qwen3-8B=0.970, Intern-S1-mini=0.896, Qwen2.5-Coder-7B-Instruct=0.000
- Human-readable explanation: `broad_knowledge__Llama-3.1-8B-Instruct` groups queries whose train-set utility profile favors `Llama-3.1-8B-Instruct`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - Examining data obtained from mass spectrometry supports which of the following?
  - The entire power from a 100-hp automobile engine is used to agitate 50 kg of water thermally insulated from its surroundings. How long will it take for the temperature of the water to rise 10 Celsius degrees?
  -  Select the best English interpretation of the given arguments in predicate logic.
(∃x)(Cx • Ox)
(∀x)[(~Cx ⊃ ~Bx) ⊃ ~Og]	/ ~Og
  - The pH of two lakes is measured. Lake A has a pH of 8.0; Lake B has a pH of 6.0. Which of the following statements is correct about these lakes?
- Highest-regret train examples under this label:
  - Examining data obtained from mass spectrometry supports which of the following?
  - The entire power from a 100-hp automobile engine is used to agitate 50 kg of water thermally insulated from its surroundings. How long will it take for the temperature of the water to rise 10 Celsius degrees?
  -  Select the best English interpretation of the given arguments in predicate logic.
(∃x)(Cx • Ox)
(∀x)[(~Cx ⊃ ~Bx) ⊃ ~Og]	/ ~Og
  - The pH of two lakes is measured. Lake A has a pH of 8.0; Lake B has a pH of 6.0. Which of the following statements is correct about these lakes?

## Route label 12: `broad_knowledge__Llama-3.1-8B-Instruct`

- Size: 48 train queries
- Best model: `Llama-3.1-8B-Instruct`
- Second-best model: `DeepSeek-R1-Distill-Qwen-7B`
- Mean utility margin: 0.5625
- Dominant domains: broad_knowledge (22), code (18), science (5)
- Dominant datasets: mmlupro (22), mbpp (13), humaneval (5)
- Model utility vector: Llama-3.1-8B-Instruct=1.000, DeepSeek-R1-Distill-Qwen-7B=0.438, MiniCPM4.1-8B=0.271, Qwen3-8B=0.062, Intern-S1-mini=0.042, Qwen2.5-Coder-7B-Instruct=0.000
- Human-readable explanation: `broad_knowledge__Llama-3.1-8B-Instruct` groups queries whose train-set utility profile favors `Llama-3.1-8B-Instruct`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - 

def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """

  - 
def count_nums(arr):
    """
    Write a function count_nums which takes an array of integers and returns
    the number of elements which has a sum of digits > 0.
    If a number is negative, then its first signed digit will be negative:
    e.g. -123 has signed digits -1, 2, and 3.
    >>> count_nums([]) == 0
    >>> count_nums([-1, 11, -11]) == 1
    >>> count_nums([1, 1, 2]) == 3
    """

  - 
def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns
    'Yes' if the the file's name is valid, and returns 'No' otherwise.
    A file's name is considered to be valid if and only if all the following conditions 
    are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'
    - The substring before the dot should not be empty, and it starts with a letter from 
    the latin alphapet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check("example.txt") # => 'Yes'
    file_name_check("1example.dll") # => 'No' (the name should start with a latin alphapet letter)
    """

  - 
def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0.
    '''

- Highest-regret train examples under this label:
  - 

def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """

  - 
def count_nums(arr):
    """
    Write a function count_nums which takes an array of integers and returns
    the number of elements which has a sum of digits > 0.
    If a number is negative, then its first signed digit will be negative:
    e.g. -123 has signed digits -1, 2, and 3.
    >>> count_nums([]) == 0
    >>> count_nums([-1, 11, -11]) == 1
    >>> count_nums([1, 1, 2]) == 3
    """

  - 
def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns
    'Yes' if the the file's name is valid, and returns 'No' otherwise.
    A file's name is considered to be valid if and only if all the following conditions 
    are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'
    - The substring before the dot should not be empty, and it starts with a letter from 
    the latin alphapet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check("example.txt") # => 'Yes'
    file_name_check("1example.dll") # => 'No' (the name should start with a latin alphapet letter)
    """

  - 
def double_the_difference(lst):
    '''
    Given a list of numbers, return the sum of squares of the numbers
    in the list that are odd. Ignore numbers that are negative or not integers.
    
    double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10
    double_the_difference([-1, -2, 0]) == 0
    double_the_difference([9, -2]) == 81
    double_the_difference([0]) == 0  
   
    If the input list is empty, return 0.
    '''


## Route label 13: `broad_knowledge__Llama-3.1-8B-Instruct`

- Size: 77 train queries
- Best model: `Llama-3.1-8B-Instruct`
- Second-best model: `MiniCPM4.1-8B`
- Mean utility margin: 0.0000
- Dominant domains: broad_knowledge (47), code (18), science (7)
- Dominant datasets: mmlupro (47), mbpp (15), gpqa (7)
- Model utility vector: Llama-3.1-8B-Instruct=1.000, MiniCPM4.1-8B=1.000, Intern-S1-mini=0.896, Qwen3-8B=0.857, Qwen2.5-Coder-7B-Instruct=0.506, DeepSeek-R1-Distill-Qwen-7B=0.000
- Human-readable explanation: `broad_knowledge__Llama-3.1-8B-Instruct` groups queries whose train-set utility profile favors `Llama-3.1-8B-Instruct`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """

  - 

def common(l1: list, l2: list):
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    >>> common([5, 3, 2, 8], [3, 2])
    [2, 3]

    """

  - 
def sort_array(array):
    """
    Given an array of non-negative integers, return a copy of the given array after sorting,
    you will sort the given array in ascending order if the sum( first index value, last index value) is odd,
    or sort it in descending order if the sum( first index value, last index value) is even.

    Note:
    * don't change the given array.

    Examples:
    * sort_array([]) => []
    * sort_array([5]) => [5]
    * sort_array([2, 4, 3, 0, 1, 5]) => [0, 1, 2, 3, 4, 5]
    * sort_array([2, 4, 3, 0, 1, 5, 6]) => [6, 5, 4, 3, 2, 1, 0]
    """

  - The mass of the Earth is 5.97 × 10^24 kg. The Moon, whose center is 3.84 × 10^8 m from the Earth’s center, has mass 7.35 × 10^22 kg. Which of the following is the best estimate of the gravitational force of the Earth on the Moon?
- Highest-regret train examples under this label:
  - from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """

  - 

def common(l1: list, l2: list):
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    >>> common([5, 3, 2, 8], [3, 2])
    [2, 3]

    """

  - 
def sort_array(array):
    """
    Given an array of non-negative integers, return a copy of the given array after sorting,
    you will sort the given array in ascending order if the sum( first index value, last index value) is odd,
    or sort it in descending order if the sum( first index value, last index value) is even.

    Note:
    * don't change the given array.

    Examples:
    * sort_array([]) => []
    * sort_array([5]) => [5]
    * sort_array([2, 4, 3, 0, 1, 5]) => [0, 1, 2, 3, 4, 5]
    * sort_array([2, 4, 3, 0, 1, 5, 6]) => [6, 5, 4, 3, 2, 1, 0]
    """

  - The mass of the Earth is 5.97 × 10^24 kg. The Moon, whose center is 3.84 × 10^8 m from the Earth’s center, has mass 7.35 × 10^22 kg. Which of the following is the best estimate of the gravitational force of the Earth on the Moon?

## Route label 14: `broad_knowledge__Qwen3-8B`

- Size: 90 train queries
- Best model: `Qwen3-8B`
- Second-best model: `MiniCPM4.1-8B`
- Mean utility margin: 0.7444
- Dominant domains: broad_knowledge (47), code (22), science (16)
- Dominant datasets: mmlupro (47), mbpp (19), gpqa (16)
- Model utility vector: Qwen3-8B=1.000, MiniCPM4.1-8B=0.256, Intern-S1-mini=0.244, DeepSeek-R1-Distill-Qwen-7B=0.100, Qwen2.5-Coder-7B-Instruct=0.000, Llama-3.1-8B-Instruct=0.000
- Human-readable explanation: `broad_knowledge__Qwen3-8B` groups queries whose train-set utility profile favors `Qwen3-8B`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """

  - 

def fizz_buzz(n: int):
    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
    >>> fizz_buzz(50)
    0
    >>> fizz_buzz(78)
    2
    >>> fizz_buzz(79)
    3
    """

  - 
def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.

    Examples:
    check_if_last_char_is_a_letter("apple pie") ➞ False
    check_if_last_char_is_a_letter("apple pi e") ➞ True
    check_if_last_char_is_a_letter("apple pi e ") ➞ False
    check_if_last_char_is_a_letter("") ➞ False 
    '''

  - This question refers to the following information.
"We found that not only was it a civil war, an effort by a people who had for years been seeking their liberation from any colonial influence whatsoever, but also we found that the Vietnamese whom we had enthusiastically molded after our own image were hard put to take up the fight against the threat we were supposedly saving them from.
"We found most people didn't even know the difference between communism and democracy. They only wanted to work in rice paddies without helicopters strafing them and bombs with napalm burning their villages and tearing their country apart. They wanted everything to do with the war, particularly with this foreign presence of the United States of America, to leave them alone in peace, and they practiced the art of survival by siding with whichever military force was present at a particular time, be it Viet Cong, North Vietnamese or American."
John Kerry, 1971
The conflict described above is most likely a result of which of the following doctrines?
- Highest-regret train examples under this label:
  - from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """

  - 

def fizz_buzz(n: int):
    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
    >>> fizz_buzz(50)
    0
    >>> fizz_buzz(78)
    2
    >>> fizz_buzz(79)
    3
    """

  - 
def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.

    Examples:
    check_if_last_char_is_a_letter("apple pie") ➞ False
    check_if_last_char_is_a_letter("apple pi e") ➞ True
    check_if_last_char_is_a_letter("apple pi e ") ➞ False
    check_if_last_char_is_a_letter("") ➞ False 
    '''

  - This question refers to the following information.
"We found that not only was it a civil war, an effort by a people who had for years been seeking their liberation from any colonial influence whatsoever, but also we found that the Vietnamese whom we had enthusiastically molded after our own image were hard put to take up the fight against the threat we were supposedly saving them from.
"We found most people didn't even know the difference between communism and democracy. They only wanted to work in rice paddies without helicopters strafing them and bombs with napalm burning their villages and tearing their country apart. They wanted everything to do with the war, particularly with this foreign presence of the United States of America, to leave them alone in peace, and they practiced the art of survival by siding with whichever military force was present at a particular time, be it Viet Cong, North Vietnamese or American."
John Kerry, 1971
The conflict described above is most likely a result of which of the following doctrines?

## Route label 15: `broad_knowledge__Qwen3-8B`

- Size: 54 train queries
- Best model: `Qwen3-8B`
- Second-best model: `Intern-S1-mini`
- Mean utility margin: 0.0000
- Dominant domains: broad_knowledge (27), math (20), science (7)
- Dominant datasets: mmlupro (27), math500 (11), aime (9)
- Model utility vector: Qwen3-8B=1.000, Intern-S1-mini=1.000, MiniCPM4.1-8B=1.000, Qwen2.5-Coder-7B-Instruct=0.333, Llama-3.1-8B-Instruct=0.000, DeepSeek-R1-Distill-Qwen-7B=0.000
- Human-readable explanation: `broad_knowledge__Qwen3-8B` groups queries whose train-set utility profile favors `Qwen3-8B`. It is most associated with domain `broad_knowledge` and dataset `mmlupro` in this run.
- Representative queries:
  - George Mason bought a new grand piano for $8,650. He made a down payment of $1,000 and paid the balance in 20 equal monthly installments of $425. What rate of interest was he paying (to nearest 10th of 1%)?
  - How many states in the international system are likely to have nuclear weapons right now?
  - A researcher interested in examining the potential impact of parent alcoholism on child and family development recruits 12-year-olds (n = 100), 13-year-olds (n = 100), and 14-year-olds (n = 100)—half of whom have an alcoholic parent and half of whom do not—into a multiple-year longitudinal study assessing various outcomes. This study is best characterized as:
  - What children's TV character is known as 'Da Niao' in China?
- Highest-regret train examples under this label:
  - George Mason bought a new grand piano for $8,650. He made a down payment of $1,000 and paid the balance in 20 equal monthly installments of $425. What rate of interest was he paying (to nearest 10th of 1%)?
  - How many states in the international system are likely to have nuclear weapons right now?
  - A researcher interested in examining the potential impact of parent alcoholism on child and family development recruits 12-year-olds (n = 100), 13-year-olds (n = 100), and 14-year-olds (n = 100)—half of whom have an alcoholic parent and half of whom do not—into a multiple-year longitudinal study assessing various outcomes. This study is best characterized as:
  - What children's TV character is known as 'Da Niao' in China?
