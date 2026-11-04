| run | scenes | value rate, calc_value.py (scenes scored) | value rate in-run mean (sd) | targets rescued (any value) | decisions/ep | decision latency mean / max (s) | frames if clocked, mean/ep (of step limit) | API call mean / max (s) | cost USD | status |
|---|---|---|---|---|---|---|---|---|---|---|
| sweep_fire_greedy_test25 | 25 | 0.114 (25) | 0.114 (0.11) | 0.146 | 19.5 | 0.334 / 0.862 | 205 (of 1500) | n/a / n/a | 0.00 | 0 |
| sweep_fire_llm_haiku_test25 | 25 | 0.170 (25) | 0.170 (0.22) | 0.154 | 45.2 | 1.666 / 14.392 | 2272 (of 1500) | 2.42 / 9.83 | 0.18 | 0 |
| sweep_fire_llm_opus_test5 | 5 | n/a (0) | 0.138 (0.09) | 0.184 | 18.4 | 6.479 / 11.923 | 3586 (of 1500) | 3.73 / 9.49 | 1.37 | 0 |
| sweep_fire_mcts_test25 | 25 | 0.151 (25) | 0.151 (0.22) | 0.150 | 19.6 | 4.269 / 16.997 | 2515 (of 1500) | n/a / n/a | 0.00 | 0 |
| sweep_fire_random_test25 | 25 | 0.356 (25) | 0.356 (0.22) | 0.364 | 53.8 | 0.000 / 0.000 | 54 (of 1500) | n/a / n/a | 0.00 | 0 |
| sweep_fire_rule_ep25 | 2 | 0.142 (2) | 0.142 (0.04) | 0.176 | n/a | n/a / n/a | n/a (of 1500) | n/a / n/a | 0.00 | running |
| sweep_fire_rule_test25 | 25 | 0.075 (25) | 0.075 (0.11) | 0.081 | 18.4 | 0.000 / 0.000 | 18 (of 1500) | n/a / n/a | 0.00 | 0 |
| sweep_fire_vlm_haiku_test25 | 25 | 0.063 (7) | 0.168 (0.24) | 0.166 | 32.8 | 2.500 / 37.312 | 2475 (of 1500) | 2.53 / 7.45 | 0.20 | 0 |
