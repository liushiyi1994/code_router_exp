# RouteCode Code Cards

These cards summarize route labels learned from train-set utility profiles. They are synthetic-pilot diagnostics, not paper claims.

## Route label 0: `data_transformation__general_8b`

- Size: 43 train queries
- Best model: `general_8b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.1728
- Dominant domains: data_transformation (13), instruction_following (12), systems_debugging (7)
- Dominant datasets: synthetic_data (13), synthetic_instructions (12), synthetic_debug (7)
- Model utility vector: general_8b=0.302, frontier_expensive=0.129, code_7b=0.124, math_7b=0.122, reasoner_13b=0.102, tiny_cheap=0.087
- Human-readable explanation: `data_transformation__general_8b` groups queries whose train-set utility profile favors `general_8b`. It is most associated with domain `data_transformation` and dataset `synthetic_data` in this run.
- Representative queries:
  - Synthetic data transformation request 9: solve a route_01 case with difficulty 0.37.
  - Synthetic instruction following request 99: solve a route_05 case with difficulty 0.40.
  - Synthetic instruction following request 100: solve a route_02 case with difficulty 0.44.
  - Synthetic data transformation request 109: solve a route_01 case with difficulty 0.43.
- Highest-regret train examples under this label:
  - Synthetic data transformation request 1769: solve a route_06 case with difficulty 0.40.
  - Synthetic instruction following request 411: solve a route_08 case with difficulty 0.48.
  - Synthetic data transformation request 1610: solve a route_06 case with difficulty 0.45.
  - Synthetic general knowledge request 592: solve a route_06 case with difficulty 0.43.

## Route label 1: `general_knowledge__tiny_cheap`

- Size: 106 train queries
- Best model: `tiny_cheap`
- Second-best model: `general_8b`
- Mean utility margin: 0.4079
- Dominant domains: general_knowledge (104), instruction_following (2)
- Dominant datasets: synthetic_easy (104), synthetic_instructions (2)
- Model utility vector: tiny_cheap=0.799, general_8b=0.391, frontier_expensive=0.305, reasoner_13b=0.258, math_7b=0.243, code_7b=0.156
- Human-readable explanation: `general_knowledge__tiny_cheap` groups queries whose train-set utility profile favors `tiny_cheap`. It is most associated with domain `general_knowledge` and dataset `synthetic_easy` in this run.
- Representative queries:
  - Synthetic general knowledge request 3: solve a route_00 case with difficulty 0.27.
  - Synthetic general knowledge request 22: solve a route_06 case with difficulty 0.34.
  - Synthetic general knowledge request 24: solve a route_06 case with difficulty 0.23.
  - Synthetic general knowledge request 31: solve a route_00 case with difficulty 0.30.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 1906: solve a route_06 case with difficulty 0.26.
  - Synthetic general knowledge request 3: solve a route_00 case with difficulty 0.27.
  - Synthetic general knowledge request 24: solve a route_06 case with difficulty 0.23.
  - Synthetic general knowledge request 31: solve a route_00 case with difficulty 0.30.

## Route label 2: `symbolic_math__code_7b`

- Size: 81 train queries
- Best model: `code_7b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.3415
- Dominant domains: symbolic_math (76), instruction_following (3), data_transformation (1)
- Dominant datasets: synthetic_math (76), synthetic_instructions (3), synthetic_data (1)
- Model utility vector: code_7b=0.632, frontier_expensive=0.290, math_7b=0.249, reasoner_13b=0.244, tiny_cheap=0.080, general_8b=0.048
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 1: solve a route_02 case with difficulty 0.34.
  - Synthetic symbolic math request 10: solve a route_08 case with difficulty 0.40.
  - Synthetic symbolic math request 23: solve a route_08 case with difficulty 0.36.
  - Synthetic symbolic math request 59: solve a route_08 case with difficulty 0.30.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 110: solve a route_11 case with difficulty 0.34.
  - Synthetic symbolic math request 1701: solve a route_10 case with difficulty 0.36.
  - Synthetic instruction following request 1330: solve a route_02 case with difficulty 0.37.
  - Synthetic symbolic math request 62: solve a route_03 case with difficulty 0.37.

## Route label 3: `general_knowledge__tiny_cheap`

- Size: 103 train queries
- Best model: `tiny_cheap`
- Second-best model: `general_8b`
- Mean utility margin: 0.2439
- Dominant domains: general_knowledge (93), symbolic_math (5), instruction_following (3)
- Dominant datasets: synthetic_easy (93), synthetic_math (5), synthetic_instructions (3)
- Model utility vector: tiny_cheap=0.608, general_8b=0.364, code_7b=0.273, reasoner_13b=0.190, math_7b=0.164, frontier_expensive=0.163
- Human-readable explanation: `general_knowledge__tiny_cheap` groups queries whose train-set utility profile favors `tiny_cheap`. It is most associated with domain `general_knowledge` and dataset `synthetic_easy` in this run.
- Representative queries:
  - Synthetic general knowledge request 21: solve a route_06 case with difficulty 0.27.
  - Synthetic general knowledge request 30: solve a route_00 case with difficulty 0.29.
  - Synthetic instruction following request 75: solve a route_00 case with difficulty 0.39.
  - Synthetic general knowledge request 87: solve a route_06 case with difficulty 0.33.
- Highest-regret train examples under this label:
  - Synthetic routine code request 542: solve a route_00 case with difficulty 0.32.
  - Synthetic general knowledge request 522: solve a route_07 case with difficulty 0.37.
  - Synthetic symbolic math request 1624: solve a route_00 case with difficulty 0.43.
  - Synthetic general knowledge request 1899: solve a route_02 case with difficulty 0.35.

## Route label 4: `instruction_following__reasoner_13b`

- Size: 134 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.4564
- Dominant domains: instruction_following (133), systems_debugging (1)
- Dominant datasets: synthetic_instructions (133), synthetic_debug (1)
- Model utility vector: reasoner_13b=0.694, frontier_expensive=0.237, tiny_cheap=0.029, general_8b=0.024, math_7b=-0.018, code_7b=-0.034
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic instruction following request 34: solve a route_04 case with difficulty 0.35.
  - Synthetic instruction following request 43: solve a route_10 case with difficulty 0.35.
  - Synthetic instruction following request 46: solve a route_04 case with difficulty 0.50.
  - Synthetic instruction following request 81: solve a route_04 case with difficulty 0.42.
- Highest-regret train examples under this label:
  - Synthetic systems debugging request 1009: solve a route_05 case with difficulty 0.44.
  - Synthetic instruction following request 34: solve a route_04 case with difficulty 0.35.
  - Synthetic instruction following request 46: solve a route_04 case with difficulty 0.50.
  - Synthetic instruction following request 81: solve a route_04 case with difficulty 0.42.

## Route label 5: `data_transformation__math_7b`

- Size: 123 train queries
- Best model: `math_7b`
- Second-best model: `reasoner_13b`
- Mean utility margin: 0.3880
- Dominant domains: data_transformation (102), instruction_following (10), symbolic_math (6)
- Dominant datasets: synthetic_data (102), synthetic_instructions (10), synthetic_math (6)
- Model utility vector: math_7b=0.608, reasoner_13b=0.220, frontier_expensive=0.112, general_8b=0.052, code_7b=0.016, tiny_cheap=0.015
- Human-readable explanation: `data_transformation__math_7b` groups queries whose train-set utility profile favors `math_7b`. It is most associated with domain `data_transformation` and dataset `synthetic_data` in this run.
- Representative queries:
  - Synthetic data transformation request 4: solve a route_03 case with difficulty 0.48.
  - Synthetic data transformation request 13: solve a route_03 case with difficulty 0.40.
  - Synthetic data transformation request 57: solve a route_03 case with difficulty 0.43.
  - Synthetic data transformation request 64: solve a route_03 case with difficulty 0.46.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 889: solve a route_03 case with difficulty 0.34.
  - Synthetic instruction following request 2225: solve a route_03 case with difficulty 0.52.
  - Synthetic instruction following request 1996: solve a route_09 case with difficulty 0.48.
  - Synthetic data transformation request 1087: solve a route_09 case with difficulty 0.43.

## Route label 6: `routine_code__general_8b`

- Size: 32 train queries
- Best model: `general_8b`
- Second-best model: `code_7b`
- Mean utility margin: 0.2832
- Dominant domains: routine_code (14), symbolic_math (7), instruction_following (5)
- Dominant datasets: synthetic_code (14), synthetic_math (7), synthetic_instructions (5)
- Model utility vector: general_8b=0.660, code_7b=0.377, frontier_expensive=0.311, reasoner_13b=0.278, math_7b=0.154, tiny_cheap=0.129
- Human-readable explanation: `routine_code__general_8b` groups queries whose train-set utility profile favors `general_8b`. It is most associated with domain `routine_code` and dataset `synthetic_code` in this run.
- Representative queries:
  - Synthetic symbolic math request 25: solve a route_07 case with difficulty 0.28.
  - Synthetic symbolic math request 65: solve a route_02 case with difficulty 0.24.
  - Synthetic routine code request 67: solve a route_11 case with difficulty 0.14.
  - Synthetic instruction following request 134: solve a route_02 case with difficulty 0.47.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 134: solve a route_02 case with difficulty 0.47.
  - Synthetic data transformation request 1740: solve a route_11 case with difficulty 0.36.
  - Synthetic instruction following request 421: solve a route_07 case with difficulty 0.21.
  - Synthetic symbolic math request 65: solve a route_02 case with difficulty 0.24.

## Route label 7: `symbolic_math__code_7b`

- Size: 88 train queries
- Best model: `code_7b`
- Second-best model: `math_7b`
- Mean utility margin: 0.4090
- Dominant domains: symbolic_math (84), general_knowledge (4)
- Dominant datasets: synthetic_math (84), synthetic_easy (4)
- Model utility vector: code_7b=0.835, math_7b=0.426, frontier_expensive=0.343, reasoner_13b=0.327, general_8b=0.263, tiny_cheap=0.179
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 14: solve a route_08 case with difficulty 0.19.
  - Synthetic symbolic math request 29: solve a route_02 case with difficulty 0.26.
  - Synthetic symbolic math request 36: solve a route_08 case with difficulty 0.32.
  - Synthetic symbolic math request 49: solve a route_08 case with difficulty 0.34.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 1763: solve a route_03 case with difficulty 0.29.
  - Synthetic symbolic math request 771: solve a route_09 case with difficulty 0.26.
  - Synthetic symbolic math request 1525: solve a route_08 case with difficulty 0.27.
  - Synthetic symbolic math request 14: solve a route_08 case with difficulty 0.19.

## Route label 8: `symbolic_math__code_7b`

- Size: 85 train queries
- Best model: `code_7b`
- Second-best model: `general_8b`
- Mean utility margin: 0.5084
- Dominant domains: symbolic_math (83), general_knowledge (2)
- Dominant datasets: synthetic_math (83), synthetic_easy (2)
- Model utility vector: code_7b=0.786, general_8b=0.278, frontier_expensive=0.260, math_7b=0.236, reasoner_13b=0.143, tiny_cheap=0.103
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 51: solve a route_02 case with difficulty 0.29.
  - Synthetic symbolic math request 136: solve a route_02 case with difficulty 0.29.
  - Synthetic symbolic math request 166: solve a route_02 case with difficulty 0.29.
  - Synthetic symbolic math request 176: solve a route_02 case with difficulty 0.27.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 51: solve a route_02 case with difficulty 0.29.
  - Synthetic symbolic math request 136: solve a route_02 case with difficulty 0.29.
  - Synthetic symbolic math request 166: solve a route_02 case with difficulty 0.29.
  - Synthetic symbolic math request 176: solve a route_02 case with difficulty 0.27.

## Route label 9: `symbolic_math__code_7b`

- Size: 85 train queries
- Best model: `code_7b`
- Second-best model: `math_7b`
- Mean utility margin: 0.5274
- Dominant domains: symbolic_math (81), instruction_following (1), systems_debugging (1)
- Dominant datasets: synthetic_math (81), synthetic_instructions (1), synthetic_debug (1)
- Model utility vector: code_7b=0.678, math_7b=0.151, general_8b=0.072, reasoner_13b=0.071, frontier_expensive=0.071, tiny_cheap=0.049
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 28: solve a route_02 case with difficulty 0.45.
  - Synthetic symbolic math request 71: solve a route_02 case with difficulty 0.42.
  - Synthetic symbolic math request 72: solve a route_02 case with difficulty 0.45.
  - Synthetic symbolic math request 105: solve a route_08 case with difficulty 0.36.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 28: solve a route_02 case with difficulty 0.45.
  - Synthetic symbolic math request 71: solve a route_02 case with difficulty 0.42.
  - Synthetic symbolic math request 72: solve a route_02 case with difficulty 0.45.
  - Synthetic symbolic math request 105: solve a route_08 case with difficulty 0.36.

## Route label 10: `instruction_following__reasoner_13b`

- Size: 121 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.4441
- Dominant domains: instruction_following (115), systems_debugging (4), general_knowledge (2)
- Dominant datasets: synthetic_instructions (115), synthetic_debug (4), synthetic_easy (2)
- Model utility vector: reasoner_13b=0.493, frontier_expensive=0.049, general_8b=0.043, tiny_cheap=0.027, math_7b=-0.043, code_7b=-0.045
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic instruction following request 0: solve a route_08 case with difficulty 0.47.
  - Synthetic instruction following request 6: solve a route_04 case with difficulty 0.46.
  - Synthetic instruction following request 16: solve a route_04 case with difficulty 0.42.
  - Synthetic instruction following request 70: solve a route_10 case with difficulty 0.45.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 297: solve a route_00 case with difficulty 0.49.
  - Synthetic instruction following request 478: solve a route_00 case with difficulty 0.51.
  - Synthetic instruction following request 1379: solve a route_08 case with difficulty 0.50.
  - Synthetic systems debugging request 2378: solve a route_05 case with difficulty 0.46.

## Route label 11: `instruction_following__reasoner_13b`

- Size: 93 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.4142
- Dominant domains: instruction_following (85), symbolic_math (5), general_knowledge (1)
- Dominant datasets: synthetic_instructions (85), synthetic_math (5), synthetic_easy (1)
- Model utility vector: reasoner_13b=0.725, frontier_expensive=0.311, general_8b=0.222, tiny_cheap=0.094, math_7b=0.086, code_7b=0.066
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic instruction following request 48: solve a route_10 case with difficulty 0.39.
  - Synthetic instruction following request 56: solve a route_04 case with difficulty 0.39.
  - Synthetic instruction following request 69: solve a route_10 case with difficulty 0.41.
  - Synthetic instruction following request 98: solve a route_04 case with difficulty 0.28.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 1994: solve a route_10 case with difficulty 0.32.
  - Synthetic instruction following request 56: solve a route_04 case with difficulty 0.39.
  - Synthetic instruction following request 69: solve a route_10 case with difficulty 0.41.
  - Synthetic instruction following request 98: solve a route_04 case with difficulty 0.28.

## Route label 12: `data_transformation__math_7b`

- Size: 86 train queries
- Best model: `math_7b`
- Second-best model: `reasoner_13b`
- Mean utility margin: 0.4491
- Dominant domains: data_transformation (76), general_knowledge (7), instruction_following (1)
- Dominant datasets: synthetic_data (76), synthetic_easy (7), synthetic_instructions (1)
- Model utility vector: math_7b=0.801, reasoner_13b=0.351, frontier_expensive=0.303, general_8b=0.181, code_7b=0.143, tiny_cheap=0.100
- Human-readable explanation: `data_transformation__math_7b` groups queries whose train-set utility profile favors `math_7b`. It is most associated with domain `data_transformation` and dataset `synthetic_data` in this run.
- Representative queries:
  - Synthetic data transformation request 12: solve a route_03 case with difficulty 0.38.
  - Synthetic instruction following request 37: solve a route_09 case with difficulty 0.39.
  - Synthetic data transformation request 76: solve a route_09 case with difficulty 0.39.
  - Synthetic data transformation request 90: solve a route_03 case with difficulty 0.35.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 869: solve a route_03 case with difficulty 0.25.
  - Synthetic general knowledge request 623: solve a route_09 case with difficulty 0.32.
  - Synthetic data transformation request 76: solve a route_09 case with difficulty 0.39.
  - Synthetic data transformation request 90: solve a route_03 case with difficulty 0.35.

## Route label 13: `systems_debugging__frontier_expensive`

- Size: 85 train queries
- Best model: `frontier_expensive`
- Second-best model: `reasoner_13b`
- Mean utility margin: 0.3898
- Dominant domains: systems_debugging (73), instruction_following (7), data_transformation (3)
- Dominant datasets: synthetic_debug (73), synthetic_instructions (7), synthetic_data (3)
- Model utility vector: frontier_expensive=0.511, reasoner_13b=0.121, general_8b=0.088, tiny_cheap=0.085, math_7b=-0.002, code_7b=-0.035
- Human-readable explanation: `systems_debugging__frontier_expensive` groups queries whose train-set utility profile favors `frontier_expensive`. It is most associated with domain `systems_debugging` and dataset `synthetic_debug` in this run.
- Representative queries:
  - Synthetic systems debugging request 11: solve a route_05 case with difficulty 0.44.
  - Synthetic systems debugging request 17: solve a route_05 case with difficulty 0.43.
  - Synthetic systems debugging request 174: solve a route_05 case with difficulty 0.41.
  - Synthetic systems debugging request 183: solve a route_05 case with difficulty 0.46.
- Highest-regret train examples under this label:
  - Synthetic systems debugging request 11: solve a route_05 case with difficulty 0.44.
  - Synthetic systems debugging request 17: solve a route_05 case with difficulty 0.43.
  - Synthetic systems debugging request 174: solve a route_05 case with difficulty 0.41.
  - Synthetic systems debugging request 183: solve a route_05 case with difficulty 0.46.

## Route label 14: `general_knowledge__tiny_cheap`

- Size: 96 train queries
- Best model: `tiny_cheap`
- Second-best model: `general_8b`
- Mean utility margin: 0.2888
- Dominant domains: general_knowledge (96)
- Dominant datasets: synthetic_easy (96)
- Model utility vector: tiny_cheap=0.817, general_8b=0.528, frontier_expensive=0.413, code_7b=0.403, reasoner_13b=0.369, math_7b=0.315
- Human-readable explanation: `general_knowledge__tiny_cheap` groups queries whose train-set utility profile favors `tiny_cheap`. It is most associated with domain `general_knowledge` and dataset `synthetic_easy` in this run.
- Representative queries:
  - Synthetic general knowledge request 5: solve a route_06 case with difficulty 0.23.
  - Synthetic general knowledge request 54: solve a route_06 case with difficulty 0.30.
  - Synthetic general knowledge request 93: solve a route_06 case with difficulty 0.16.
  - Synthetic general knowledge request 121: solve a route_01 case with difficulty 0.25.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 1276: solve a route_07 case with difficulty 0.16.
  - Synthetic general knowledge request 467: solve a route_10 case with difficulty 0.20.
  - Synthetic general knowledge request 1286: solve a route_11 case with difficulty 0.24.
  - Synthetic general knowledge request 1387: solve a route_11 case with difficulty 0.12.

## Route label 15: `general_knowledge__tiny_cheap`

- Size: 79 train queries
- Best model: `tiny_cheap`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.0458
- Dominant domains: general_knowledge (58), symbolic_math (9), instruction_following (7)
- Dominant datasets: synthetic_easy (58), synthetic_math (9), synthetic_instructions (7)
- Model utility vector: tiny_cheap=0.485, frontier_expensive=0.439, reasoner_13b=0.314, general_8b=0.241, code_7b=0.206, math_7b=0.146
- Human-readable explanation: `general_knowledge__tiny_cheap` groups queries whose train-set utility profile favors `tiny_cheap`. It is most associated with domain `general_knowledge` and dataset `synthetic_easy` in this run.
- Representative queries:
  - Synthetic general knowledge request 15: solve a route_00 case with difficulty 0.25.
  - Synthetic general knowledge request 19: solve a route_06 case with difficulty 0.27.
  - Synthetic general knowledge request 39: solve a route_00 case with difficulty 0.29.
  - Synthetic general knowledge request 74: solve a route_10 case with difficulty 0.30.
- Highest-regret train examples under this label:
  - Synthetic systems debugging request 2290: solve a route_11 case with difficulty 0.34.
  - Synthetic symbolic math request 2043: solve a route_11 case with difficulty 0.33.
  - Synthetic general knowledge request 2212: solve a route_04 case with difficulty 0.25.
  - Synthetic symbolic math request 1700: solve a route_11 case with difficulty 0.23.
