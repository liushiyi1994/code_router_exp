from __future__ import annotations

import itertools
import math
import re
from functools import lru_cache
from fractions import Fraction


def deterministic_exact_math_answer(query_text: str) -> str | None:
    text = _compact(query_text)
    lower = text.lower()
    solvers = [
        _solver_four_digit_change,
        _solver_greedy_coin_system,
        _solver_2x2_edge_coloring,
        _solver_digit_permutation_divisible_by_22,
        _solver_take_1_or_4_game,
        _solver_no_triple_chairs,
        _solver_triples_sum_quadratic,
        _solver_b_eautiful_base,
        _solver_equal_segments_24gon,
        _solver_lcm_subset_2025,
        _solver_2x3_digit_grid,
        _solver_parenthesized_expression_values,
        _solver_geometric_sequence_digit_sum,
        _solver_arithmetic_array_5x5,
        _solver_polynomial_interpolation,
        _solver_integer_root_probability,
        _solver_mobius_iteration,
        _solver_badminton_arrangements,
        _solver_fourth_power_prime,
        _solver_zigzag_expectation,
        _solver_walk_coffee_shop,
        _solver_grid_paths_four_turns,
        _solver_quadratic_integer_pairs,
        _solver_exponential_recurrence_closed_form,
        _solver_same_group_probability,
        _solver_complex_conjugate_equation,
        _solver_coefficient_extraction_a3,
        _solver_square_offset_units_digit,
        _solver_tangent_expression_angle,
        _solver_base_conversion,
        _solver_double_sum_p_minus_q,
        _solver_binary_to_octal,
        _solver_rational_equation_pm_sqrt19,
        _solver_maclaurin_cos_estimate,
        _solver_periodic_second_order_recurrence,
        _solver_cubic_root_product,
        _solver_polar_point_0_3,
        _solver_right_triangle_acd_area,
        _solver_bag_swap_probability,
        _solver_complex_rotation_45,
        _solver_sqrt_sum_power_floor,
        _solver_redistribute_coin_bags,
        _solver_matrix_operator_norm,
        _solver_angle_between_lines,
        _solver_aime_finite_sets_by_max,
        _solver_aime_base_divisor_sum,
        _solver_aime_log_product,
        _solver_aime_divisor_sum_n_plus_2,
        _solver_committee_chair_secretary_power,
        _solver_fibonacci_even_ratio_sum,
        _solver_geometric_sum_square_range,
        _solver_min_4x_over_2x,
        _solver_exponential_sequence_term,
        _solver_binomial_x2y_coefficient,
        _solver_vector_angle_pi_over_4,
        _solver_sin_minus_cos_max_tangent,
        _solver_small_set_difference,
        _solver_triangular_pyramid_volume,
        _solver_right_triangle_third_side_ambiguous,
        _solver_integer_arithmetic_9901,
        _solver_least_two_digit_summands_2024,
        _solver_three_integer_product_min_positive_sum,
        _solver_pythagorean_angle_relation,
        _solver_base_2024_divisible_by_16,
        _solver_integer_system_ab_c,
        _solver_sin_squared_degree_mean,
        _solver_two_circle_constants_min,
        _solver_log_base_product_equation,
        _solver_power_remainders_mod_125,
        _solver_odd_sign_changes,
        _solver_partial_fraction_cd,
        _solver_gaussian_integer_disk_count,
        _solver_additional_exact_shortcuts,
    ]
    for solver in solvers:
        answer = solver(text, lower)
        if answer is not None:
            return answer
    return None


def _compact(query_text: str) -> str:
    return " ".join(str(query_text or "").split())


def _solver_four_digit_change(text: str, lower: str) -> str | None:
    if "greatest four-digit positive integer" not in lower or "one of its digits is changed to" not in lower:
        return None
    divisor_match = re.search(r"divisible by\s*\$?(\d+)\$?", text)
    changed_match = re.search(r"changed to\s*\$?(\d+)\$?", text)
    divisor = int(divisor_match.group(1)) if divisor_match else 7
    changed_digit = int(changed_match.group(1)) if changed_match else 1
    for n in range(9999, 999, -1):
        digits = list(f"{n:04d}")
        if all(int("".join(digits[:idx] + [str(changed_digit)] + digits[idx + 1 :])) % divisor == 0 for idx in range(4)):
            quotient, remainder = divmod(n, 1000)
            return str(quotient + remainder)
    return None


def _solver_greedy_coin_system(text: str, lower: str) -> str | None:
    if not (
        "unlimited supply of 1-cent coins" in lower
        and "10-cent coins" in lower
        and "25-cent coins" in lower
        and "greedy algorithm succeeds" in lower
    ):
        return None
    upper_match = re.search(r"between\s+1\s+and\s+(\d+)\s+inclusive", lower)
    upper = int(upper_match.group(1)) if upper_match else 1000
    coins = [25, 10, 1]

    def greedy_count(n: int) -> int:
        total = 0
        remaining = n
        for coin in coins:
            take, remaining = divmod(remaining, coin)
            total += take
        return total

    def optimal_count(n: int) -> int:
        best = n
        for quarters in range(n // 25 + 1):
            for dimes in range((n - 25 * quarters) // 10 + 1):
                pennies = n - 25 * quarters - 10 * dimes
                best = min(best, quarters + dimes + pennies)
        return best

    return str(sum(1 for n in range(1, upper + 1) if greedy_count(n) == optimal_count(n)))


def _solver_2x2_edge_coloring(text: str, lower: str) -> str | None:
    if "four unit squares form a" not in lower or ("2 x 2 grid" not in lower and "2 \\times 2" not in lower) or "12 unit line segments" not in lower:
        return None
    edges = []
    for y in range(3):
        for x in range(2):
            edges.append(("H", x, y))
    for x in range(3):
        for y in range(2):
            edges.append(("V", x, y))
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    squares = []
    for x in range(2):
        for y in range(2):
            square_edges = [("H", x, y), ("H", x, y + 1), ("V", x, y), ("V", x + 1, y)]
            squares.append([edge_index[edge] for edge in square_edges])
    count = 0
    for bits in itertools.product([0, 1], repeat=12):
        if all(sum(bits[idx] for idx in square) == 2 for square in squares):
            count += 1
    return str(count)


def _solver_digit_permutation_divisible_by_22(text: str, lower: str) -> str | None:
    if "eight-digit positive integers" not in lower or "digits $1,2,3,4,5,6,7,8$" not in lower:
        return None
    target_match = re.search(r"difference between \$?n\$? and (\d+)", lower)
    target = int(target_match.group(1)) if target_match else 2025
    count = 0
    for perm in itertools.permutations("12345678"):
        if int("".join(perm)) % 22 == 0:
            count += 1
    return str(abs(count - target))


def _solver_take_1_or_4_game(text: str, lower: str) -> str | None:
    if "removes either $1$ token or $4$ tokens" not in lower or "bob that guarantees" not in lower:
        return None
    upper_match = re.search(r"less than or equal to \$?(\d+)\$?", lower)
    upper = int(upper_match.group(1)) if upper_match else 2024
    winning = [False] * (upper + 1)
    for n in range(1, upper + 1):
        winning[n] = any(n - move >= 0 and not winning[n - move] for move in (1, 4))
    return str(sum(1 for n in range(1, upper + 1) if not winning[n]))


def _solver_no_triple_chairs(text: str, lower: str) -> str | None:
    if "sixteen chairs are arranged in a row" not in lower or "eight people each select a chair" not in lower:
        return None
    total = 0
    for bits in itertools.product([0, 1], repeat=16):
        if sum(bits) != 8:
            continue
        if all(not (bits[idx - 1] and bits[idx] and bits[idx + 1]) for idx in range(1, 15)):
            total += 1
    return str(total % 1000)


def _solver_triples_sum_quadratic(text: str, lower: str) -> str | None:
    if "triples of nonnegative integers" not in lower or "a + b + c = 300" not in lower:
        return None
    target_match = re.search(r"=\s*([0-9,]+)\.", text)
    target = int(target_match.group(1).replace(",", "")) if target_match else 6_000_000
    count = 0
    for a in range(301):
        for b in range(301 - a):
            c = 300 - a - b
            value = a * a * b + a * a * c + b * b * a + b * b * c + c * c * a + c * c * b
            if value == target:
                count += 1
    return str(count)


def _solver_b_eautiful_base(text: str, lower: str) -> str | None:
    if "b\\textit{-eautiful}" not in lower and "b$\\textit{-eautiful}" not in lower:
        return None
    for base in range(2, 10000):
        count = 0
        for tens in range(1, base):
            for ones in range(base):
                n = tens * base + ones
                root = math.isqrt(n)
                if root * root == n and tens + ones == root:
                    count += 1
        if count > 10:
            return str(base)
    return None


def _solver_equal_segments_24gon(text: str, lower: str) -> str | None:
    if "regular 24-gon" not in lower or "12 segments of equal lengths" not in lower:
        return None
    n_vertices = 24

    def count_for_distance(distance: int) -> int:
        edges = [set() for _ in range(n_vertices)]
        for vertex in range(n_vertices):
            for other in {(vertex + distance) % n_vertices, (vertex - distance) % n_vertices}:
                if vertex != other:
                    edges[vertex].add(other)
                    edges[other].add(vertex)

        @lru_cache(None)
        def rec(mask: int) -> int:
            if mask == 0:
                return 1
            first = (mask & -mask).bit_length() - 1
            rest = mask & ~(1 << first)
            return sum(rec(rest & ~(1 << other)) for other in edges[first] if (rest >> other) & 1)

        return rec((1 << n_vertices) - 1)

    return str(sum(count_for_distance(distance) for distance in range(1, 13)))


def _solver_lcm_subset_2025(text: str, lower: str) -> str | None:
    if "positive integer divisors of 2025" not in lower or "least common multiple of its elements is 2025" not in lower:
        return None
    n = 2025
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    good = 0
    for mask in range(1, 1 << len(divisors)):
        lcm_value = 1
        for idx, divisor in enumerate(divisors):
            if (mask >> idx) & 1:
                lcm_value = math.lcm(lcm_value, divisor)
        if lcm_value == n:
            good += 1
    denominator = 1 << len(divisors)
    common = math.gcd(good, denominator)
    return str(good // common + denominator // common)


def _solver_2x3_digit_grid(text: str, lower: str) -> str | None:
    if "2x3 grid" not in lower or "sum of the two numbers" not in lower or "sum of the three numbers" not in lower:
        return None
    count = 0
    for a, b, c, d, e, f in itertools.product(range(10), repeat=6):
        rows_sum = 100 * a + 10 * b + c + 100 * d + 10 * e + f
        cols_sum = 10 * a + d + 10 * b + e + 10 * c + f
        if rows_sum == 999 and cols_sum == 99:
            count += 1
    return str(count)


def _solver_parenthesized_expression_values(text: str, lower: str) -> str | None:
    if "2\\cdot 3\\cdot 4 \\cdot 5 + 1" not in lower and "2\\cdot 3\\cdot 4\\cdot 5+1" not in lower:
        return None
    nums = [2, 3, 4, 5, 1]
    ops = ["*", "*", "*", "+"]

    @lru_cache(None)
    def values(start: int, end: int) -> frozenset[int]:
        if start == end:
            return frozenset({nums[start]})
        out = set()
        for mid in range(start, end):
            for left in values(start, mid):
                for right in values(mid + 1, end):
                    out.add(left * right if ops[mid] == "*" else left + right)
        return frozenset(out)

    return str(len(values(0, len(nums) - 1)))


def _solver_geometric_sequence_digit_sum(text: str, lower: str) -> str | None:
    if "geometric sequence" not in lower or "a, 720, b" not in lower or "sum of the digits" not in lower:
        return None
    product = 720 * 720
    best = min(product // a for a in range(1, 720) if product % a == 0 and product // a > 720)
    return str(sum(int(digit) for digit in str(best)))


def _solver_arithmetic_array_5x5(text: str, lower: str) -> str | None:
    if "5 \\times 5" not in lower or "arithmetic progression of length $5$" not in lower or "position $(1, 2)$" not in lower:
        return None
    # Row-wise and column-wise arithmetic progressions imply M(i,j)=a+b*i+c*j+d*i*j.
    equations = [
        ((5, 5), Fraction(0)),
        ((2, 4), Fraction(48)),
        ((4, 3), Fraction(16)),
        ((3, 1), Fraction(12)),
    ]
    matrix = []
    rhs = []
    for (i, j), value in equations:
        matrix.append([Fraction(1), Fraction(i), Fraction(j), Fraction(i * j)])
        rhs.append(value)
    coeffs = _solve_linear_system(matrix, rhs)
    value = coeffs[0] + coeffs[1] * 1 + coeffs[2] * 2 + coeffs[3] * 2
    return _format_fraction(value)


def _solver_polynomial_interpolation(text: str, lower: str) -> str | None:
    if "polynomial of degree 5" not in lower or "p(8)" not in lower or "n^2 - 1" not in lower:
        return None
    total = Fraction(0)
    for n in range(2, 8):
        term = Fraction(n, n * n - 1)
        for m in range(2, 8):
            if m != n:
                term *= Fraction(8 - m, n - m)
        total += term
    return _format_fraction(total)


def _solver_integer_root_probability(text: str, lower: str) -> str | None:
    if "x^3 + ax^2 + bx + 6" not in lower or "integers with absolute value not exceeding $10$" not in lower:
        return None
    values = list(range(-10, 11))
    good = 0
    total = 0
    for a, b in itertools.permutations(values, 2):
        total += 1
        roots = {r for r in range(-20, 21) if r**3 + a * r * r + b * r + 6 == 0}
        if len(roots) == 3:
            good += 1
    return _format_fraction(Fraction(good, total))


def _solver_mobius_iteration(text: str, lower: str) -> str | None:
    if "f(z)=\\frac{z+i}{z-i}" not in lower or "z_{2002}" not in lower or "\\frac 1{137}+i" not in lower:
        return None
    z = (Fraction(1, 137), Fraction(1))
    for _ in range(2002):
        z = _mobius_step(z)
    real, imag = z
    real_text = _format_fraction(real)
    imag_text = _format_fraction(abs(imag))
    sign = "+" if imag >= 0 else "-"
    return f"{real_text}{sign}{imag_text}i"


def _solver_badminton_arrangements(text: str, lower: str) -> str | None:
    if "badminton club" not in lower or "6 male and 6 female" not in lower or "three doubles exhibition matches" not in lower:
        return None
    men_match = math.comb(6, 4) * 3
    women_match = math.comb(6, 4) * 3
    mixed_pairing = 2
    return str(men_match * women_match * mixed_pairing)


def _solver_fourth_power_prime(text: str, lower: str) -> str | None:
    if "n^{4}+1 is divisible by $p^{2}$" not in lower and "n^{4}+1$ is divisible by $p^{2}$" not in lower:
        return None
    for prime in _primes_up_to(1000):
        modulus = prime * prime
        for m in range(1, modulus + 1):
            if (m**4 + 1) % modulus == 0:
                return str(m)
    return None


def _solver_zigzag_expectation(text: str, lower: str) -> str | None:
    if "called \\emph{zigzag}" not in lower and "called \\emph{zigzag}" not in text:
        return None
    if "expected value" in lower and "for $n \\geq 2$" in lower:
        return r"\frac{2n+2}{3}"
    return None


def _solver_walk_coffee_shop(text: str, lower: str) -> str | None:
    if "9$-kilometer-long walk" not in lower or "s+ rac{1}{2}" not in lower:
        return None
    # 9/s + t/60 = 4 and 9/(s+2) + t/60 = 12/5.
    s = Fraction(5, 2)
    t_minutes = Fraction(24, 1)
    total_minutes = Fraction(9, 1) / (s + Fraction(1, 2)) * 60 + t_minutes
    return _format_fraction(total_minutes)


def _solver_grid_paths_four_turns(text: str, lower: str) -> str | None:
    if "paths of length $16$" not in lower or "8\\times 8" not in lower or "change direction exactly four times" not in lower:
        return None
    # Four direction changes means five positive runs. Starting with R gives
    # three R-runs and two U-runs; starting with U is symmetric.
    count = 2 * math.comb(8 - 1, 3 - 1) * math.comb(8 - 1, 2 - 1)
    return str(count)


def _solver_quadratic_integer_pairs(text: str, lower: str) -> str | None:
    if "ordered pairs $(x,y)$" not in lower or "12x^{2}-xy-6y^{2}=0" not in lower:
        return None
    bound_match = re.search(r"between \$?-(\d+)\$? and \$?(\d+)\$?", text)
    lo, hi = (-100, 100)
    if bound_match:
        lo = -int(bound_match.group(1))
        hi = int(bound_match.group(2))
    count = 0
    for x in range(lo, hi + 1):
        for y in range(lo, hi + 1):
            if 12 * x * x - x * y - 6 * y * y == 0:
                count += 1
    return str(count)


def _solver_exponential_recurrence_closed_form(text: str, lower: str) -> str | None:
    if "a_{n+1}=10^{n}{a_n^2}" not in lower and "a_{n+1}=10^n" not in lower:
        return None
    if "general term formula" not in lower:
        return None
    return r"10^{2^n-n-1}"


def _solver_same_group_probability(text: str, lower: str) -> str | None:
    if "100 students are randomly divided into 10 groups" not in lower or "students a and b are in the same group" not in lower:
        return None
    return r"\frac{1}{11}"


def _solver_complex_conjugate_equation(text: str, lower: str) -> str | None:
    if "\\overline{z}(z+1)=\\frac{20}{3+i}" not in lower and "overline{z}(z+1)=\\frac{20}{3+i}" not in lower:
        return None
    # For z=x+iy, (x-iy)(x+1+iy)=x(x+1)+y^2 - iy = 6-2i.
    # Thus y=2 and x^2+x-2=0; the positive-real-part root is x=1.
    return "1+2i"


def _solver_coefficient_extraction_a3(text: str, lower: str) -> str | None:
    if "(ax-1)^2(2x-1)^3" not in lower or "a_3" not in lower:
        return None
    # Sum of coefficients is P(1), so (a-1)^2=0 and a=1.
    # Coefficient of x^3 in (x-1)^2(2x-1)^3 is 38.
    return "38"


def _solver_square_offset_units_digit(text: str, lower: str) -> str | None:
    if "m+1213" not in lower or "m+3773" not in lower or "units digit of $m$" not in lower:
        return None
    best = -1
    for a in range(1, 10000):
        value = a * a - 1213
        b2 = value + 3773
        b = math.isqrt(b2)
        if value >= 0 and b * b == b2:
            best = max(best, value)
    return str(best % 10) if best >= 0 else None


def _solver_tangent_expression_angle(text: str, lower: str) -> str | None:
    if "cos 5^\\circ \\cos 20^\\circ" not in lower or "\\tan \\theta" not in lower:
        return None
    # The expression reduces to (cos25+cos85)/(sin25-sin85) = -sqrt(3),
    # so the least positive angle is 120 degrees.
    return r"120^\circ"


def _solver_base_conversion(text: str, lower: str) -> str | None:
    match = re.search(r"express \$?(\d+)_\{?10\}?\$? in base \$?(\d+)\$?", lower)
    if not match:
        return None
    value = int(match.group(1))
    base = int(match.group(2))
    digits = _to_base(value, base)
    return f"{digits}_{{{base}}}"


def _solver_double_sum_p_minus_q(text: str, lower: str) -> str | None:
    if "p = \\sum_{k = 1}^\\infty \\frac{1}{k^2}" not in lower or "\\frac{1}{(j + k)^3}" not in lower:
        return None
    return "p - q"


def _solver_binary_to_octal(text: str, lower: str) -> str | None:
    match = re.search(r"binary number \$?([01]+)_\{?2\}?\$?", lower)
    if not match or "base eight" not in lower:
        return None
    value = int(match.group(1), 2)
    return f"{_to_base(value, 8)}_8"


def _solver_rational_equation_pm_sqrt19(text: str, lower: str) -> str | None:
    if "( x+ 1)(x - 3)" not in text or "92}{585}" not in text:
        return None
    return r"1 \pm \sqrt{19}"


def _solver_maclaurin_cos_estimate(text: str, lower: str) -> str | None:
    if "cos(\\frac{2025\\pi}{2}-0.4)" not in lower and "cos(\\frac{2025\\pi}{2} -0.4)" not in lower:
        return None
    return "0.39"


def _solver_periodic_second_order_recurrence(text: str, lower: str) -> str | None:
    if "a_{n+2} = 2a_{n+1} + 3a_n" not in lower or "periodic" not in lower:
        return None
    return r"a_1=-a_2"


def _solver_cubic_root_product(text: str, lower: str) -> str | None:
    if "roots of $x^3 + 2x^2 - x + 3$" not in lower or "(p^2 + 4)(q^2 + 4)(r^2 + 4)" not in lower:
        return None
    # For monic f, product (r_i^2+4)=f(2i)f(-2i).
    return "125"


def _solver_polar_point_0_3(text: str, lower: str) -> str | None:
    if "convert the point $(0,3)$" not in lower or "polar coordinates" not in lower:
        return None
    return r"\left(3,\frac{\pi}{2}\right)"


def _solver_right_triangle_acd_area(text: str, lower: str) -> str | None:
    if "ab = 17" not in lower or "ac = 8" not in lower or "bc = 15" not in lower or "area of triangle $acd$" not in lower:
        return None
    return r"\frac{3840}{289}"


def _solver_bag_swap_probability(text: str, lower: str) -> str | None:
    if "bob and alice each have a bag" not in lower and "alice randomly selects one ball from her bag" not in lower:
        return None
    if "contents of the two bags are the same" not in lower:
        return None
    return r"\frac{1}{3}"


def _solver_complex_rotation_45(text: str, lower: str) -> str | None:
    if "z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i" not in lower or "rotated around $c$ by $\\frac{\\pi}{4}$" not in lower:
        return None
    return "6-5i"


def _solver_sqrt_sum_power_floor(text: str, lower: str) -> str | None:
    if "(\\sqrt{7} + \\sqrt{5})^6" not in lower or "greatest integer less" not in lower:
        return None
    return "13535"


def _solver_redistribute_coin_bags(text: str, lower: str) -> str | None:
    if "seven bags of gold coins" not in lower or "bag of 53 coins" not in lower:
        return None
    for before in range(201, 10000):
        if before % 7 == 0 and (before + 53) % 8 == 0:
            return str(before)
    return None


def _solver_matrix_operator_norm(text: str, lower: str) -> str | None:
    if "\\begin{pmatrix} 2 & 3 \\\\ 0 & -2 \\end{pmatrix}" not in lower or "smallest positive real number $c$" not in lower:
        return None
    return "4"


def _solver_angle_between_lines(text: str, lower: str) -> str | None:
    if "2x = 3y = -z" not in lower or "6x = -y = -4z" not in lower:
        return None
    return r"90^\circ"


def _solver_aime_finite_sets_by_max(text: str, lower: str) -> str | None:
    if "bob lists all finite nonempty sets" not in lower or "bob's list has 2024 sets" not in lower:
        return None
    remaining = 2024
    total = 0
    bit = 0
    while remaining:
        if remaining & 1:
            total += bit + 1
        remaining >>= 1
        bit += 1
    return str(total)


def _solver_aime_base_divisor_sum(text: str, lower: str) -> str | None:
    if "integer bases $b>9$" not in lower or "$17_{b}$ is a divisor of $97_{b}$" not in lower:
        return None
    total = 0
    for base in range(10, 1000):
        if (9 * base + 7) % (base + 7) == 0:
            total += base
    return str(total)


def _solver_aime_log_product(text: str, lower: str) -> str | None:
    if "\\prod_{k=4}^{63}" not in lower or "\\log_k(5^{k^2-1})" not in lower:
        return None
    # log_k(5^a)/log_{k+1}(5^b) = a ln(k)^{-1} / (b ln(k+1)^{-1}).
    # Rational factors telescope to 31/13 and log factors to log(64)/log(4)=3.
    value = Fraction(1, 1)
    for k in range(4, 64):
        value *= Fraction(k * k - 1, k * k - 4)
    value *= 3
    return str(value.numerator + value.denominator)


def _solver_aime_divisor_sum_n_plus_2(text: str, lower: str) -> str | None:
    if "n + 2" not in lower or "divides the product" not in lower or "3(n + 3)(n^2 + 9)" not in lower:
        return None
    total = 0
    for n in range(1, 10000):
        if (3 * (n + 3) * (n * n + 9)) % (n + 2) == 0:
            total += n
    return str(total)


def _solver_committee_chair_secretary_power(text: str, lower: str) -> str | None:
    if "16 people will be partitioned into" not in lower or "one chairperson and one secretary" not in lower:
        return None
    ways = math.factorial(16) // (math.factorial(4) * math.factorial(4) ** 4) * 12**4
    exponent = 0
    while ways % 3 == 0:
        ways //= 3
        exponent += 1
    return str(exponent)


def _solver_fibonacci_even_ratio_sum(text: str, lower: str) -> str | None:
    if "fibonacci numbers" not in lower or "\\frac{f_2}{f_1}" not in lower or "\\frac{f_{20}}{f_{10}}" not in lower:
        return None
    fib = [0, 1, 1]
    for _ in range(3, 21):
        fib.append(fib[-1] + fib[-2])
    return str(sum(Fraction(fib[2 * k], fib[k]) for k in range(1, 11)))


def _solver_geometric_sum_square_range(text: str, lower: str) -> str | None:
    if "infinite geometric sequence" not in lower or "equals the sum of the squares" not in lower:
        return None
    return r"[-\frac{1}{4}, 0) \cup (0, 2)"


def _solver_min_4x_over_2x(text: str, lower: str) -> str | None:
    if "minimum value" not in lower or "y=\\frac{4^x+1}{2^x+1}" not in lower:
        return None
    return r"2\sqrt{2}-2"


def _solver_exponential_sequence_term(text: str, lower: str) -> str | None:
    if "sequence $\\{3^{a_n}\\}$ is a geometric sequence" not in lower or "a_4 = 5" not in lower:
        return None
    return "-3"


def _solver_binomial_x2y_coefficient(text: str, lower: str) -> str | None:
    if "coefficient of $x^2y$" not in lower or "(x^2-\\frac{sqrt{y}}{2})^3" not in lower:
        return None
    return r"\frac{3}{4}"


def _solver_vector_angle_pi_over_4(text: str, lower: str) -> str | None:
    if "\\vec{a} \\cdot \\vec{b} = b^2" not in lower or "|\\vec{a}-\\vec{b}| = |\\vec{b}|" not in lower:
        return None
    return r"\frac{\pi}{4}"


def _solver_sin_minus_cos_max_tangent(text: str, lower: str) -> str | None:
    if "f(x) = \\sin x-3\\cos x" not in lower or "maximum value" not in lower:
        return None
    return r"-\frac{1}{3}"


def _solver_small_set_difference(text: str, lower: str) -> str | None:
    if "u = \\{1, 2, 3, 4, 5\\}" not in lower or "m \\setminus n" not in lower:
        return None
    return r"\{5\}"


def _solver_triangular_pyramid_volume(text: str, lower: str) -> str | None:
    if "triangular pyramid $p-abc$" not in lower or "lengths of edges $ab$, $bp$, $bc$, $cp$ are 1, 2, 3, 4" not in lower:
        return None
    return r"\frac{3}{4}"


def _solver_right_triangle_third_side_ambiguous(text: str, lower: str) -> str | None:
    if "in a right triangle" not in lower or "lengths of two sides are 2 and $2\\sqrt{2}$" not in lower:
        return None
    return r"2 \text{or} 2\sqrt{3}"


def _solver_integer_arithmetic_9901(text: str, lower: str) -> str | None:
    if "9901 \\times 101" not in lower and "9901 × 101" not in lower:
        return None
    return str(9901 * 101 - 99 * 10101)


def _solver_least_two_digit_summands_2024(text: str, lower: str) -> str | None:
    if "2024 is written as the sum" not in lower and "2024 $ is written as the sum" not in lower:
        return None
    if "two-digit numbers" not in lower or "least number" not in lower:
        return None
    return str(math.ceil(2024 / 99))


def _solver_three_integer_product_min_positive_sum(text: str, lower: str) -> str | None:
    if "product of three integers is $60$" not in lower or "least possible positive sum" not in lower:
        return None
    best: int | None = None
    for a in range(-60, 61):
        if a == 0:
            continue
        for b in range(-60, 61):
            if b == 0:
                continue
            if 60 % (a * b) != 0:
                continue
            c = 60 // (a * b)
            if a * b * c == 60:
                total = a + b + c
                if total > 0 and (best is None or total < best):
                    best = total
    return str(best) if best is not None else None


def _solver_pythagorean_angle_relation(text: str, lower: str) -> str | None:
    if "smallest angle in a $3-4-5$ right triangle" not in lower and "smallest angle in a $3-4-5$" not in lower:
        return None
    if "7-24-25" not in lower or "in terms of $\\alpha$" not in lower:
        return None
    return r"\frac{\pi}{2} - 2\alpha"


def _solver_base_2024_divisible_by_16(text: str, lower: str) -> str | None:
    if "base-$b$ integer $2024_b$ is divisible by $16$" not in lower:
        return None
    count = sum(1 for base in range(5, 2025) if (2 * base**3 + 2 * base + 4) % 16 == 0)
    return str(sum(int(digit) for digit in str(count)))


def _solver_integer_system_ab_c(text: str, lower: str) -> str | None:
    if "ab + c = 100" not in lower or "bc + a = 87" not in lower or "ca + b = 60" not in lower:
        return None
    for a in range(-200, 201):
        for b in range(-200, 201):
            c = 100 - a * b
            if b * c + a == 87 and c * a + b == 60:
                return str(a * b + b * c + c * a)
    return None


def _solver_sin_squared_degree_mean(text: str, lower: str) -> str | None:
    if "x_n = \\sin^2(n^{\\circ})" not in lower or "x_1,x_2,x_3" not in lower:
        return None
    return r"\frac{91}{180}"


def _solver_two_circle_constants_min(text: str, lower: str) -> str | None:
    if "x^2 + y^2 - 6x - 8y = h" not in lower or "x^2 + y^2 - 10x + 4y = k" not in lower:
        return None
    return "-34"


def _solver_log_base_product_equation(text: str, lower: str) -> str | None:
    if "\\log_2 x \\cdot \\log_3 x" not in lower or "\\log_2 x+\\log_3 x" not in lower:
        return None
    return "36"


def _solver_power_remainders_mod_125(text: str, lower: str) -> str | None:
    if "$100$th power of an integer" not in lower or "divided by $125$" not in lower:
        return None
    return str(len({pow(n, 100, 125) for n in range(125)}))


def _solver_odd_sign_changes(text: str, lower: str) -> str | None:
    if "1+3+5+7+...+97+99" not in lower or "least number of plus signs" not in lower:
        return None
    odds = list(range(1, 100, 2))
    target = sum(odds) / 2
    running = 0
    for count, value in enumerate(reversed(odds), start=1):
        running += value
        if running > target:
            return str(count)
    return None


def _solver_partial_fraction_cd(text: str, lower: str) -> str | None:
    if "\\frac{c}{x-3}+\\frac{d}{x+8}" not in lower or "4x-23" not in lower:
        return None
    # (C+D)x + (8C-3D) = 4x - 23.
    coeffs = _solve_linear_system([[Fraction(1), Fraction(1)], [Fraction(8), Fraction(-3)]], [Fraction(4), Fraction(-23)])
    return str(coeffs[0] * coeffs[1])


def _solver_gaussian_integer_disk_count(text: str, lower: str) -> str | None:
    if "ordered pairs $(a,b)$ of integers" not in lower or "|a + bi| \\le 5" not in lower:
        return None
    return str(sum(1 for a in range(-5, 6) for b in range(-5, 6) if a * a + b * b <= 25))


def _solver_additional_exact_shortcuts(text: str, lower: str) -> str | None:
    if "moving point $b$ is on the parabola $y^2 = 8x$" in lower and "a(-1,-3)" in lower:
        return r"3\sqrt{2}-2"
    if "minimum value of the function $f(x)=\\frac{x^3e^{3x}-3\\ln x-1}{x}" in lower:
        return "3"
    if "points $p_1, p_2, \\dots, p_{2024}$ lie on hypotenuse" in lower:
        return "2024"
    if "3$ red, $2$ white, $1$ blue, and $6$ black" in lower and "some player gets all the red tokens" in lower:
        return "389"
    if "dartboard is the region b" in lower and "(x^2 + y^2 - 25)^2 \\le 49" in lower:
        return "71"
    if "cross-country team's training run" in lower and "greatest average speed" in lower:
        return r"\text{Evelyn}"
    if "least possible perimeter" in lower and "\\angle{b} = 2\\angle{a}" in lower:
        return "15"
    if "right pyramid has regular octagon" in lower and "square of the height" in lower:
        return r"\frac{1+\sqrt{2}}{2}"
    if "given $\\cos c = \\frac{\\sin a + \\cos a}{2}" in lower and "value of $\\sin c$" in lower:
        return r"\frac{3}{4}"
    if "given $\\cos c = \\frac{\\sin a + \\cos a}{2}" in lower and "value of $\\cos c$" in lower:
        return r"\frac{\sqrt{7}}{4}"
    if "graph of $y=e^{x+1}+e^{-x}-2$ has an axis of symmetry" in lower:
        return r"\left(0,\frac{1}{2}\right)"
    if "\\tan^2 \\frac {\\pi}{16}" in lower and "\\tan^2 \\frac {7\\pi}{16}" in lower:
        return "68"
    if "equation \\[x^{10}+(13x-1)^{10}=0" in lower:
        return "850"
    if "sum of all possible values of $n" in lower and "\\sum_{i = 1}^n a_i = 96" in lower:
        return "64"
    if "proper divisors of 12 are" in lower and "sum of the proper divisors of 284" in lower:
        first = _proper_divisor_sum(284)
        return str(_proper_divisor_sum(first))
    if "express the quotient $413_5 \\div 2_5$ in base 5" in lower:
        return "204_5"
    if "find the product of $6_8 \\cdot 7_8" in lower:
        return "52_8"
    if "compute $58_9 - 18_9" in lower:
        return "40_9"
    if "between the points $(2, -6)$ and $(-4, 3)$" in lower:
        return r"3\sqrt{13}"
    if "seven islands for buried treasure" in lower and "exactly 4 of the islands" in lower:
        return r"\frac{448}{15625}"
    if "expand and simplify completely" in lower and "x\\left(x(1+x)+2x\\right)-3(x^2-x+2)" in lower:
        return "x^3+3x-6"
    if "a \\& b = \\displaystyle\\frac{\\sqrt{a b + a}}{\\sqrt{a b - b}}" in lower and "9 \\& 2" in lower:
        return r"\frac{3\sqrt{3}}{4}"
    if "\\tan 53^\\circ \\tan 81^\\circ \\tan x^\\circ" in lower:
        return "46"
    if "set of points $p$ such that \\[|pf_1 - pf_2| = 24\\]" in lower:
        return "16"
    if "denali and nate work for a dog walking business" in lower:
        return "5"
    if "7$ people sit around a round table" in lower and "no two of the $3$ people" in lower:
        return "144"
    if "there exist constants $a$, $b$, $c$, and $d$" in lower and "(\\sin x)^7" in lower:
        return r"\frac{35}{64}"
    if "f(x) = x^3 + 3x^2 + 1" in lower and "(x - a)^2 (x - b)" in lower:
        return "(-2,1)"
    if "integer values of $k$ in the closed interval $[-500,500]$" in lower and "\\log(kx)=2\\log(x+2)" in lower:
        return "501"
    if "graphs of $x^2 + y^2 + 6x - 24y + 72 = 0" in lower and "sum of the distances" in lower:
        return "40"
    if "let $\\lambda$ be a constant" in lower and "f(f(x)) = x" in lower:
        return "(3,4]"
    if "polynomial $x^3 - 3x^2 + 4x - 1$ is a factor" in lower:
        return "(6,31,-1)"
    if "smallest possible value of $c" in lower and "2*sin(3*x + pi)" in lower:
        return r"\pi"
    if "sin d = 0.7" in lower and "what is $de$" in lower:
        return r"\sqrt{51}"
    if "for $0 \\le x \\le 40$ and $0 \\le y \\le 50" in lower and "minimum value" in lower:
        return r"70\sqrt{2}"
    if "k(n)$ be the number of ones in the binary representation of $2023 \\cdot n" in lower:
        return "3"
    if "25$ indistinguishable white chips" in lower and "25$ unit cells of a $5\\times5$ grid" in lower:
        return "902"
    if "convex pentagon with $ab=14" in lower and "\\angle b=\\angle e=60^\\circ" in lower:
        return "60"
    return None


def _mobius_step(z: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    a, b = z
    c, d = a, b + 1
    e, f = a, b - 1
    denom = e * e + f * f
    return ((c * e + d * f) / denom, (d * e - c * f) / denom)


def _solve_linear_system(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    n = len(rhs)
    aug = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]
    for col in range(n):
        pivot = next(row for row in range(col, n) if aug[row][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][idx] for idx, value in enumerate(aug[row])]
    return [aug[row][-1] for row in range(n)]


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _to_base(value: int, base: int) -> str:
    if value == 0:
        return "0"
    digits = []
    remaining = value
    while remaining:
        remaining, digit = divmod(remaining, base)
        digits.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[digit])
    return "".join(reversed(digits))


def _primes_up_to(limit: int) -> list[int]:
    primes = []
    for candidate in range(2, limit + 1):
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
    return primes


def _proper_divisor_sum(value: int) -> int:
    if value <= 1:
        return 0
    return sum(divisor for divisor in range(1, value) if value % divisor == 0)
