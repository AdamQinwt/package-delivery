#!/usr/bin/env python
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog

import argparse
from easydict import EasyDict as edict
from src.solver import Solver
from src.evaluator import Evaluator
from src.config import cfg, merge_a_into_b
from src.graph import Info, Hub

parser = argparse.ArgumentParser(description="Package delivery problem solver")
parser.add_argument("--problem_id", dest="PROBLEM_ID",
                    help="problem_id", type=int, required=True)
parser.add_argument("--order_num", dest="NUM_ORDERS",
                    help="the number of orders to parse", type=int, default=2400)
parser.add_argument("--process_num", dest="NUM_PROCESSES",
                    help="the number of processes", type=int, default=16)
parser.add_argument("--weight_amount", dest="WEIGHT_AMOUNT",
                    help="the weight of the amountCost in the objective function", type=float, default=15)
parser.add_argument("--weight_time", dest="WEIGHT_TIME",
                    help="the weight of the timeCost in the objective function", type=float, default=1)
parser.add_argument("--hub_cost_const", dest="HUB_BUILT_COST_CONST",
                    help="the constant part of cost of building a hub", type=int, default=3000)
parser.add_argument("--hub_cost_vary", dest="HUB_BUILT_COST_VARY",
                    help="the variable part cost of building a hub", type=float, default=2.0)
parser.add_argument("--hub_ratio", dest="HUB_UNIT_COST_RATIO",
                    help="the cutoff of unit price of a hub", type=float, default=0.7)
parser.add_argument("--hub_capacity", dest="HUB_CAPACITY",
                    help="the capacity of the hub(in prob3)", type=int, default=500)

args = parser.parse_args()

arg_dict = edict(vars(args))
merge_a_into_b(arg_dict, cfg)
print("Using the config:")
print(cfg)

solver = Solver()
evaluator = Evaluator()

solver.solve(problem_id=args.PROBLEM_ID)
# disable the tune mode but plot the distribution, see the outputs for more detail
evaluator.evaluate(prob_id=args.PROBLEM_ID, tune=False, plot=True)

