# RouteCode Code Cards

These cards summarize route labels learned from train-set utility profiles. They are synthetic-pilot diagnostics, not paper claims.

## Route label 0: `instruction_following__reasoner_13b`

- Size: 27 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.3378
- Dominant domains: instruction_following (10), general_knowledge (7), symbolic_math (7)
- Dominant datasets: synthetic_instructions (10), synthetic_easy (7), synthetic_math (7)
- Model utility vector: reasoner_13b=0.579, frontier_expensive=0.241, general_8b=0.210, tiny_cheap=0.140, code_7b=0.128, math_7b=0.081
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic general knowledge request 74: solve a route_10 case with difficulty 0.30.
  - Synthetic instruction following request 159: solve a route_10 case with difficulty 0.28.
  - Synthetic systems debugging request 245: solve a route_10 case with difficulty 0.42.
  - Synthetic instruction following request 259: solve a route_10 case with difficulty 0.62.
- Highest-regret train examples under this label:
  - Synthetic data transformation request 1130: solve a route_10 case with difficulty 0.42.
  - Synthetic symbolic math request 1994: solve a route_10 case with difficulty 0.32.
  - Synthetic symbolic math request 1465: solve a route_10 case with difficulty 0.33.
  - Synthetic general knowledge request 74: solve a route_10 case with difficulty 0.30.

## Route label 1: `symbolic_math__code_7b`

- Size: 102 train queries
- Best model: `code_7b`
- Second-best model: `general_8b`
- Mean utility margin: 0.4815
- Dominant domains: symbolic_math (97), routine_code (2), general_knowledge (2)
- Dominant datasets: synthetic_math (97), synthetic_code (2), synthetic_easy (2)
- Model utility vector: code_7b=0.723, general_8b=0.241, math_7b=0.231, frontier_expensive=0.219, reasoner_13b=0.169, tiny_cheap=0.087
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 1: solve a route_02 case with difficulty 0.34.
  - Synthetic symbolic math request 28: solve a route_02 case with difficulty 0.45.
  - Synthetic symbolic math request 29: solve a route_02 case with difficulty 0.26.
  - Synthetic symbolic math request 51: solve a route_02 case with difficulty 0.29.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 786: solve a route_02 case with difficulty 0.41.
  - Synthetic symbolic math request 2336: solve a route_02 case with difficulty 0.43.
  - Synthetic symbolic math request 29: solve a route_02 case with difficulty 0.26.
  - Synthetic symbolic math request 51: solve a route_02 case with difficulty 0.29.

## Route label 2: `routine_code__general_8b`

- Size: 27 train queries
- Best model: `general_8b`
- Second-best model: `code_7b`
- Mean utility margin: 0.2638
- Dominant domains: routine_code (9), general_knowledge (7), instruction_following (4)
- Dominant datasets: synthetic_code (9), synthetic_easy (7), synthetic_instructions (4)
- Model utility vector: general_8b=0.616, code_7b=0.352, frontier_expensive=0.282, reasoner_13b=0.244, tiny_cheap=0.163, math_7b=0.097
- Human-readable explanation: `routine_code__general_8b` groups queries whose train-set utility profile favors `general_8b`. It is most associated with domain `routine_code` and dataset `synthetic_code` in this run.
- Representative queries:
  - Synthetic data transformation request 9: solve a route_01 case with difficulty 0.37.
  - Synthetic data transformation request 109: solve a route_01 case with difficulty 0.43.
  - Synthetic general knowledge request 121: solve a route_01 case with difficulty 0.25.
  - Synthetic general knowledge request 393: solve a route_01 case with difficulty 0.33.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 971: solve a route_01 case with difficulty 0.41.
  - Synthetic general knowledge request 2196: solve a route_01 case with difficulty 0.17.
  - Synthetic general knowledge request 1371: solve a route_01 case with difficulty 0.21.
  - Synthetic instruction following request 1632: solve a route_01 case with difficulty 0.40.

## Route label 3: `instruction_following__reasoner_13b`

- Size: 94 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.3946
- Dominant domains: instruction_following (85), general_knowledge (5), symbolic_math (3)
- Dominant datasets: synthetic_instructions (85), synthetic_easy (5), synthetic_math (3)
- Model utility vector: reasoner_13b=0.645, frontier_expensive=0.251, tiny_cheap=0.086, general_8b=0.061, code_7b=0.036, math_7b=0.025
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic instruction following request 16: solve a route_04 case with difficulty 0.42.
  - Synthetic instruction following request 34: solve a route_04 case with difficulty 0.35.
  - Synthetic instruction following request 56: solve a route_04 case with difficulty 0.39.
  - Synthetic instruction following request 77: solve a route_04 case with difficulty 0.42.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 106: solve a route_04 case with difficulty 0.38.
  - Synthetic general knowledge request 2212: solve a route_04 case with difficulty 0.25.
  - Synthetic general knowledge request 1262: solve a route_04 case with difficulty 0.39.
  - Synthetic instruction following request 1608: solve a route_04 case with difficulty 0.45.

## Route label 4: `symbolic_math__code_7b`

- Size: 76 train queries
- Best model: `code_7b`
- Second-best model: `math_7b`
- Mean utility margin: 0.4553
- Dominant domains: symbolic_math (72), instruction_following (2), systems_debugging (1)
- Dominant datasets: synthetic_math (72), synthetic_instructions (2), synthetic_debug (1)
- Model utility vector: code_7b=0.693, math_7b=0.238, frontier_expensive=0.196, reasoner_13b=0.155, tiny_cheap=0.090, general_8b=0.051
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 10: solve a route_08 case with difficulty 0.40.
  - Synthetic symbolic math request 59: solve a route_08 case with difficulty 0.30.
  - Synthetic symbolic math request 66: solve a route_08 case with difficulty 0.22.
  - Synthetic symbolic math request 89: solve a route_08 case with difficulty 0.33.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 10: solve a route_08 case with difficulty 0.40.
  - Synthetic symbolic math request 59: solve a route_08 case with difficulty 0.30.
  - Synthetic symbolic math request 66: solve a route_08 case with difficulty 0.22.
  - Synthetic symbolic math request 89: solve a route_08 case with difficulty 0.33.

## Route label 5: `systems_debugging__frontier_expensive`

- Size: 64 train queries
- Best model: `frontier_expensive`
- Second-best model: `general_8b`
- Mean utility margin: 0.3530
- Dominant domains: systems_debugging (38), symbolic_math (9), general_knowledge (9)
- Dominant datasets: synthetic_debug (38), synthetic_math (9), synthetic_easy (9)
- Model utility vector: frontier_expensive=0.563, general_8b=0.210, tiny_cheap=0.193, reasoner_13b=0.157, code_7b=0.074, math_7b=0.051
- Human-readable explanation: `systems_debugging__frontier_expensive` groups queries whose train-set utility profile favors `frontier_expensive`. It is most associated with domain `systems_debugging` and dataset `synthetic_debug` in this run.
- Representative queries:
  - Synthetic routine code request 67: solve a route_11 case with difficulty 0.14.
  - Synthetic symbolic math request 110: solve a route_11 case with difficulty 0.34.
  - Synthetic systems debugging request 217: solve a route_11 case with difficulty 0.43.
  - Synthetic systems debugging request 276: solve a route_11 case with difficulty 0.45.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 1387: solve a route_11 case with difficulty 0.12.
  - Synthetic general knowledge request 981: solve a route_11 case with difficulty 0.44.
  - Synthetic instruction following request 700: solve a route_11 case with difficulty 0.43.
  - Synthetic routine code request 67: solve a route_11 case with difficulty 0.14.

## Route label 6: `general_knowledge__tiny_cheap`

- Size: 169 train queries
- Best model: `tiny_cheap`
- Second-best model: `general_8b`
- Mean utility margin: 0.2870
- Dominant domains: general_knowledge (150), instruction_following (8), systems_debugging (4)
- Dominant datasets: synthetic_easy (150), synthetic_instructions (8), synthetic_debug (4)
- Model utility vector: tiny_cheap=0.692, general_8b=0.405, frontier_expensive=0.275, reasoner_13b=0.259, math_7b=0.194, code_7b=0.187
- Human-readable explanation: `general_knowledge__tiny_cheap` groups queries whose train-set utility profile favors `tiny_cheap`. It is most associated with domain `general_knowledge` and dataset `synthetic_easy` in this run.
- Representative queries:
  - Synthetic general knowledge request 3: solve a route_00 case with difficulty 0.27.
  - Synthetic general knowledge request 15: solve a route_00 case with difficulty 0.25.
  - Synthetic general knowledge request 30: solve a route_00 case with difficulty 0.29.
  - Synthetic general knowledge request 31: solve a route_00 case with difficulty 0.30.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 1335: solve a route_00 case with difficulty 0.43.
  - Synthetic systems debugging request 2341: solve a route_00 case with difficulty 0.43.
  - Synthetic general knowledge request 1618: solve a route_00 case with difficulty 0.45.
  - Synthetic routine code request 542: solve a route_00 case with difficulty 0.32.

## Route label 7: `general_knowledge__tiny_cheap`

- Size: 180 train queries
- Best model: `tiny_cheap`
- Second-best model: `general_8b`
- Mean utility margin: 0.3316
- Dominant domains: general_knowledge (164), symbolic_math (5), instruction_following (5)
- Dominant datasets: synthetic_easy (164), synthetic_math (5), synthetic_instructions (5)
- Model utility vector: tiny_cheap=0.692, general_8b=0.361, code_7b=0.309, frontier_expensive=0.288, reasoner_13b=0.265, math_7b=0.248
- Human-readable explanation: `general_knowledge__tiny_cheap` groups queries whose train-set utility profile favors `tiny_cheap`. It is most associated with domain `general_knowledge` and dataset `synthetic_easy` in this run.
- Representative queries:
  - Synthetic general knowledge request 5: solve a route_06 case with difficulty 0.23.
  - Synthetic general knowledge request 19: solve a route_06 case with difficulty 0.27.
  - Synthetic general knowledge request 21: solve a route_06 case with difficulty 0.27.
  - Synthetic general knowledge request 22: solve a route_06 case with difficulty 0.34.
- Highest-regret train examples under this label:
  - Synthetic data transformation request 2268: solve a route_06 case with difficulty 0.37.
  - Synthetic data transformation request 1125: solve a route_06 case with difficulty 0.45.
  - Synthetic symbolic math request 725: solve a route_06 case with difficulty 0.31.
  - Synthetic systems debugging request 1929: solve a route_06 case with difficulty 0.40.

## Route label 8: `instruction_following__reasoner_13b`

- Size: 158 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.4807
- Dominant domains: instruction_following (158)
- Dominant datasets: synthetic_instructions (158)
- Model utility vector: reasoner_13b=0.653, frontier_expensive=0.172, general_8b=0.100, tiny_cheap=0.018, math_7b=-0.017, code_7b=-0.045
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic instruction following request 43: solve a route_10 case with difficulty 0.35.
  - Synthetic instruction following request 48: solve a route_10 case with difficulty 0.39.
  - Synthetic instruction following request 69: solve a route_10 case with difficulty 0.41.
  - Synthetic instruction following request 70: solve a route_10 case with difficulty 0.45.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 43: solve a route_10 case with difficulty 0.35.
  - Synthetic instruction following request 48: solve a route_10 case with difficulty 0.39.
  - Synthetic instruction following request 69: solve a route_10 case with difficulty 0.41.
  - Synthetic instruction following request 70: solve a route_10 case with difficulty 0.45.

## Route label 9: `systems_debugging__frontier_expensive`

- Size: 59 train queries
- Best model: `frontier_expensive`
- Second-best model: `tiny_cheap`
- Mean utility margin: 0.2910
- Dominant domains: systems_debugging (38), general_knowledge (8), instruction_following (6)
- Dominant datasets: synthetic_debug (38), synthetic_easy (8), synthetic_instructions (6)
- Model utility vector: frontier_expensive=0.481, tiny_cheap=0.190, reasoner_13b=0.163, general_8b=0.114, math_7b=0.029, code_7b=0.005
- Human-readable explanation: `systems_debugging__frontier_expensive` groups queries whose train-set utility profile favors `frontier_expensive`. It is most associated with domain `systems_debugging` and dataset `synthetic_debug` in this run.
- Representative queries:
  - Synthetic systems debugging request 11: solve a route_05 case with difficulty 0.44.
  - Synthetic systems debugging request 17: solve a route_05 case with difficulty 0.43.
  - Synthetic instruction following request 99: solve a route_05 case with difficulty 0.40.
  - Synthetic systems debugging request 174: solve a route_05 case with difficulty 0.41.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 2217: solve a route_05 case with difficulty 0.32.
  - Synthetic general knowledge request 1656: solve a route_05 case with difficulty 0.31.
  - Synthetic instruction following request 1583: solve a route_05 case with difficulty 0.39.
  - Synthetic symbolic math request 1407: solve a route_05 case with difficulty 0.37.

## Route label 10: `symbolic_math__code_7b`

- Size: 65 train queries
- Best model: `code_7b`
- Second-best model: `math_7b`
- Mean utility margin: 0.4725
- Dominant domains: symbolic_math (54), instruction_following (6), general_knowledge (4)
- Dominant datasets: synthetic_math (54), synthetic_instructions (6), synthetic_easy (4)
- Model utility vector: code_7b=0.758, math_7b=0.285, general_8b=0.282, frontier_expensive=0.275, reasoner_13b=0.193, tiny_cheap=0.129
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic symbolic math request 65: solve a route_02 case with difficulty 0.24.
  - Synthetic symbolic math request 71: solve a route_02 case with difficulty 0.42.
  - Synthetic instruction following request 100: solve a route_02 case with difficulty 0.44.
  - Synthetic symbolic math request 103: solve a route_02 case with difficulty 0.25.
- Highest-regret train examples under this label:
  - Synthetic data transformation request 1871: solve a route_02 case with difficulty 0.28.
  - Synthetic instruction following request 134: solve a route_02 case with difficulty 0.47.
  - Synthetic instruction following request 1330: solve a route_02 case with difficulty 0.37.
  - Synthetic instruction following request 1758: solve a route_02 case with difficulty 0.51.

## Route label 11: `instruction_following__reasoner_13b`

- Size: 73 train queries
- Best model: `reasoner_13b`
- Second-best model: `frontier_expensive`
- Mean utility margin: 0.3847
- Dominant domains: instruction_following (73)
- Dominant datasets: synthetic_instructions (73)
- Model utility vector: reasoner_13b=0.595, frontier_expensive=0.210, tiny_cheap=0.060, general_8b=0.056, math_7b=0.008, code_7b=0.005
- Human-readable explanation: `instruction_following__reasoner_13b` groups queries whose train-set utility profile favors `reasoner_13b`. It is most associated with domain `instruction_following` and dataset `synthetic_instructions` in this run.
- Representative queries:
  - Synthetic instruction following request 6: solve a route_04 case with difficulty 0.46.
  - Synthetic instruction following request 46: solve a route_04 case with difficulty 0.50.
  - Synthetic instruction following request 81: solve a route_04 case with difficulty 0.42.
  - Synthetic instruction following request 96: solve a route_04 case with difficulty 0.50.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 1064: solve a route_04 case with difficulty 0.47.
  - Synthetic instruction following request 1069: solve a route_04 case with difficulty 0.43.
  - Synthetic instruction following request 81: solve a route_04 case with difficulty 0.42.
  - Synthetic instruction following request 46: solve a route_04 case with difficulty 0.50.

## Route label 12: `data_transformation__math_7b`

- Size: 120 train queries
- Best model: `math_7b`
- Second-best model: `reasoner_13b`
- Mean utility margin: 0.3546
- Dominant domains: data_transformation (97), instruction_following (8), general_knowledge (7)
- Dominant datasets: synthetic_data (97), synthetic_instructions (8), synthetic_easy (7)
- Model utility vector: math_7b=0.664, reasoner_13b=0.310, frontier_expensive=0.163, general_8b=0.107, code_7b=0.071, tiny_cheap=0.055
- Human-readable explanation: `data_transformation__math_7b` groups queries whose train-set utility profile favors `math_7b`. It is most associated with domain `data_transformation` and dataset `synthetic_data` in this run.
- Representative queries:
  - Synthetic data transformation request 4: solve a route_03 case with difficulty 0.48.
  - Synthetic data transformation request 12: solve a route_03 case with difficulty 0.38.
  - Synthetic data transformation request 13: solve a route_03 case with difficulty 0.40.
  - Synthetic data transformation request 57: solve a route_03 case with difficulty 0.43.
- Highest-regret train examples under this label:
  - Synthetic general knowledge request 881: solve a route_03 case with difficulty 0.42.
  - Synthetic instruction following request 1908: solve a route_07 case with difficulty 0.42.
  - Synthetic instruction following request 2178: solve a route_03 case with difficulty 0.40.
  - Synthetic instruction following request 501: solve a route_03 case with difficulty 0.37.

## Route label 13: `data_transformation__general_8b`

- Size: 25 train queries
- Best model: `general_8b`
- Second-best model: `code_7b`
- Mean utility margin: 0.3155
- Dominant domains: data_transformation (6), general_knowledge (6), instruction_following (4)
- Dominant datasets: synthetic_data (6), synthetic_easy (6), synthetic_instructions (4)
- Model utility vector: general_8b=0.538, code_7b=0.222, reasoner_13b=0.198, frontier_expensive=0.196, math_7b=0.168, tiny_cheap=0.140
- Human-readable explanation: `data_transformation__general_8b` groups queries whose train-set utility profile favors `general_8b`. It is most associated with domain `data_transformation` and dataset `synthetic_data` in this run.
- Representative queries:
  - Synthetic symbolic math request 25: solve a route_07 case with difficulty 0.28.
  - Synthetic data transformation request 169: solve a route_07 case with difficulty 0.47.
  - Synthetic data transformation request 355: solve a route_07 case with difficulty 0.34.
  - Synthetic instruction following request 421: solve a route_07 case with difficulty 0.21.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 421: solve a route_07 case with difficulty 0.21.
  - Synthetic data transformation request 1661: solve a route_07 case with difficulty 0.38.
  - Synthetic general knowledge request 431: solve a route_07 case with difficulty 0.38.
  - Synthetic symbolic math request 25: solve a route_07 case with difficulty 0.28.

## Route label 14: `data_transformation__math_7b`

- Size: 102 train queries
- Best model: `math_7b`
- Second-best model: `reasoner_13b`
- Mean utility margin: 0.4260
- Dominant domains: data_transformation (79), symbolic_math (9), general_knowledge (7)
- Dominant datasets: synthetic_data (79), synthetic_math (9), synthetic_easy (7)
- Model utility vector: math_7b=0.671, reasoner_13b=0.245, frontier_expensive=0.213, general_8b=0.123, code_7b=0.102, tiny_cheap=0.061
- Human-readable explanation: `data_transformation__math_7b` groups queries whose train-set utility profile favors `math_7b`. It is most associated with domain `data_transformation` and dataset `synthetic_data` in this run.
- Representative queries:
  - Synthetic instruction following request 37: solve a route_09 case with difficulty 0.39.
  - Synthetic data transformation request 76: solve a route_09 case with difficulty 0.39.
  - Synthetic data transformation request 132: solve a route_09 case with difficulty 0.35.
  - Synthetic data transformation request 172: solve a route_09 case with difficulty 0.39.
- Highest-regret train examples under this label:
  - Synthetic symbolic math request 930: solve a route_09 case with difficulty 0.31.
  - Synthetic systems debugging request 1329: solve a route_09 case with difficulty 0.59.
  - Synthetic symbolic math request 1043: solve a route_09 case with difficulty 0.23.
  - Synthetic instruction following request 1996: solve a route_09 case with difficulty 0.48.

## Route label 15: `symbolic_math__code_7b`

- Size: 99 train queries
- Best model: `code_7b`
- Second-best model: `math_7b`
- Mean utility margin: 0.4317
- Dominant domains: symbolic_math (89), general_knowledge (5), instruction_following (4)
- Dominant datasets: synthetic_math (89), synthetic_easy (5), synthetic_instructions (4)
- Model utility vector: code_7b=0.725, math_7b=0.293, frontier_expensive=0.266, reasoner_13b=0.258, tiny_cheap=0.115, general_8b=0.110
- Human-readable explanation: `symbolic_math__code_7b` groups queries whose train-set utility profile favors `code_7b`. It is most associated with domain `symbolic_math` and dataset `synthetic_math` in this run.
- Representative queries:
  - Synthetic instruction following request 0: solve a route_08 case with difficulty 0.47.
  - Synthetic symbolic math request 14: solve a route_08 case with difficulty 0.19.
  - Synthetic symbolic math request 23: solve a route_08 case with difficulty 0.36.
  - Synthetic symbolic math request 36: solve a route_08 case with difficulty 0.32.
- Highest-regret train examples under this label:
  - Synthetic instruction following request 313: solve a route_08 case with difficulty 0.62.
  - Synthetic instruction following request 0: solve a route_08 case with difficulty 0.47.
  - Synthetic general knowledge request 753: solve a route_08 case with difficulty 0.14.
  - Synthetic general knowledge request 2334: solve a route_08 case with difficulty 0.36.
