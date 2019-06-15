import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATA = edict()
__C.DATA.ROOT_DIR = "../data"
__C.DATA.VEHICLE_CSV = osp.join(__C.DATA.ROOT_DIR, "vehicle.csv")
__C.DATA.DISTANCE_CSV = osp.join(__C.DATA.ROOT_DIR, "distance.csv")
__C.DATA.COMMODITY_CSV = osp.join(__C.DATA.ROOT_DIR, "commodity.csv")
__C.DATA.ORDER_CSV = osp.join(__C.DATA.ROOT_DIR, "order.csv")

__C.MODEL = edict()
__C.MODEL.ROOT_DIR = "../model"
__C.MODEL.GRAPH_PATH = osp.join(__C.MODEL.ROOT_DIR, "graph.pkl")
__C.MODEL.GRAPH_STATION_PATH = osp.join(__C.MODEL.ROOT_DIR, "graph_station.pkl")
__C.MODEL.GRAPH_SMALL_PATH = osp.join(__C.MODEL.ROOT_DIR, "graph_small.pkl")

__C.PROB = edict()
__C.PROB.ROOT_DIR = "../output"
__C.PROB.PROBLEM1_PATH = osp.join(__C.PROB.ROOT_DIR, "prob1")
__C.PROB.PROBLEM2_PATH = osp.join(__C.PROB.ROOT_DIR, "prob2")
__C.PROB.PROBLEM3_PATH = osp.join(__C.PROB.ROOT_DIR, "prob3")
__C.PROB.PROBLEM4_PATH = osp.join(__C.PROB.ROOT_DIR, "prob4")
__C.PROB.PROBLEM_LIST = [__C.PROB.PROBLEM1_PATH, __C.PROB.PROBLEM2_PATH, __C.PROB.PROBLEM3_PATH, __C.PROB.PROBLEM4_PATH]

__C.PARAM = edict()
# weight function: weight_amount * cost + weight_time * time + constant_c
__C.PARAM.WEIGHT_AMOUNT = 15
__C.PARAM.WEIGHT_TIME = 1
__C.PARAM.WEIGHT_TIME_EMERGENCY = 2

__C.PARAM.HUB_CAPACITY = 1000
__C.PARAM.NUM_ORDERS = 2400
# number of processes
__C.PARAM.NUM_PROCESSES = 16
# cost of building a hub
__C.PARAM.HUB_BUILT_COST_CONST = 3000
__C.PARAM.HUB_BUILT_COST_VARY = 300
# ratio of the cutoff for the hub
__C.PARAM.HUB_UNIT_COST_RATIO = 0.7

__C.EVALUATION = edict()
__C.EVALUATION.ROOT_DIR = "../evaluation"
# for tuning the parameters
__C.EVALUATION.WEIGHT_TIME_FIELD = np.round(np.arange(0.3, 2, 0.3), decimals=2)
__C.EVALUATION.WEIGHT_AMOUNT_FIELD = np.round(15 * np.arange(0.3, 2, 0.3), decimals=2)
# for sensitivity test
# for problem3
__C.EVALUATION.HUB_CAPACITY_FIELD = np.arange(500, 2500, 500)
# for problem2
__C.EVALUATION.HUB_RATIO_FIELD = np.round(np.arange(0.6, 1.1, 0.1), decimals=3)
__C.EVALUATION.HUB_COST_CONST_FIELD = np.arange(1000, 10000, 2000)
__C.EVALUATION.HUB_COST_VARY_FIELD = np.arange(100, 700, 100)

# rarely be changed
__C.NUM_CITIES = 656
__C.COMMODITY_TYPES = {"Liquid": 0, "Metal": 1, "Inflammable Products": 2, "Food": 3, "Electronics": 4,
        "Big Furniture": 5, "Plastic": 6, "Glass": 7}
__C.VEHICLES = ['Plane', 'Train', 'Ship', 'Truck']
# constraint(e.g. liquid and inflammable products can not be transmitted by the plane)
__C.CONSTRAINTS = [(0, 0), (2, 0)]
# some hubs reject glass or inflammable products
__C.HUB_REJECTS = [2, 7]
# large number that will be used several times
__C.INT_MAX = 10000000
# a small number that will be used several times
__C.INT_MIN = -10000000


def merge_a_into_b(a, b):
    """ merge the config of two edicts
    """
    for k, v in a.items():
        if v is None:
            continue
        if k in b.PARAM:
            b.PARAM[k] = v
        else:
            print("Found new key:{}".format(k))
