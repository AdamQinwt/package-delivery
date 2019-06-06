#!/usr/bin/env python
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog

import re
import multiprocessing
import numpy as np
import pandas as pd
import os.path as osp
import pickle as pkl
import warnings
from tqdm import trange
from utils import std2sec, sec2std
warnings.filterwarnings('ignore')

NUM_PROCESSES = 8
NUM_CITIES = 656
DATA_DIR = "../data"
DIST_CSV_PATH = osp.join(DATA_DIR, "distance.csv")
ORDER_CSV_PATH = osp.join(DATA_DIR, "order.csv")
COMMODITY_CSV_PATH = osp.join(DATA_DIR, "commodity.csv")
VEHICLE_CSV_PATH = osp.join(DATA_DIR, "vehicle.csv")
GRAPH_SAVE_PATH = osp.join(DATA_DIR, "graph.pkl")
VEHICLES = ['Plane', 'Ship', 'Truck', 'Train']


class Graph:
    def __init__(self):
        self.vertices = set([])
        self.neighbours = {}
        self.infos = [[[] for _ in range(NUM_CITIES+1)] for _ in range(NUM_CITIES+1)]
        self._load_data()

    def get_vertices(self):
        return self.vertices

    def get_neighbours(self, src):
        if src not in self.vertices:
            raise KeyError("Error! {0} not in the vertex set!".format(src))
        return self.neighbours[src]

    def find_all_paths(self, start, end, num_transit=1, depth=0, path=[]):
        """ find all the path(i.e. between two cities) in the graph(i.e. the city network)
        :param start: source to start exploration
        :param end: destination
        :param num_transit: how many transit cities are allowed
        :param path: path already found
        :param depth: the depth of recursion
        :return: if not found, return none, otherwise return the path
        """
        path = path + [start]
        depth = depth + 1
        if depth > num_transit + 2:
            return []
        if start == end:
            return [path]
        if start not in self.vertices or end not in self.vertices:
            return []
        paths = []

        for node in self.neighbours[start]:
            if node not in path:
                new_paths = self.find_all_paths(node, end, num_transit, depth, path)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths

    def find_path(self, start, end, num_transit=1, depth=0, path=[]):
        """ find a path(i.e. between two cities) in the graph(i.e. the city network)
        :param start: source to start exploration
        :param end: destination
        :param num_transit: how many transit cities are allowed
        :param path: path already found
        :param depth: the depth of recursion
        :return: if not found, return none, otherwise return the path
        """
        path = path + [start]
        depth = depth + 1
        if depth > num_transit + 2:
            return None
        if start == end:
            return path

        if start not in self.vertices or end not in self.vertices:
            return None
        for node in self.neighbours[start]:
            if node not in path:
                new_path = self.find_path(node, end, num_transit, depth, path)
                if new_path:
                    return new_path
        return None

    def _load_data(self):
        """ construct the graph using the vehicle.csv
        note that multi-processing can not help us here
        :return:
        """
        dist = np.array(pd.read_csv(DIST_CSV_PATH))
        vehicle = pd.read_csv(VEHICLE_CSV_PATH)
        if osp.exists(GRAPH_SAVE_PATH):
            with open(GRAPH_SAVE_PATH, 'rb') as f:
                data = pkl.load(f)
            self.vertices = data['vertices']
            self.neighbours = data['neighbours']
            self.infos = data['infos']
            print("Load graph from {0} successfully.".format(GRAPH_SAVE_PATH))
            return
        print("Processing the graph...")
        t = trange(len(vehicle), desc="Graph construction", leave=True)
        for i in t:
            t.set_description("records %d" % i)
            rec = vehicle.loc[i]
            src, dest = rec['IndexOfDepartureCity'], rec['IndexOfArrivalCity']
            rec['dist'] = dist[src - 1, dest - 1]
            if src not in self.vertices:
                self.vertices.add(src)
            if dest not in self.vertices:
                self.vertices.add(dest)
            if src not in self.neighbours.keys():
                self.neighbours[src] = set([])
            self.neighbours[src].add(dest)
            self.infos[src][dest].append(rec)
            if i % 10000 == 0 and i > 0:
                with open(osp.join(DATA_DIR, "graph" + str(i) + ".pkl"), "wb") as f:
                    data = {'vertices': self.vertices, 'neighbours': self.neighbours, 'infos': self.infos, 'checkpoint': i}
                    pkl.dump(data, f)
                    print("Checkpoint %d saved." % i)

        print("Finish processing!")
        with open(GRAPH_SAVE_PATH, 'wb') as f:
            data = {'vertices': self.vertices, 'neighbours': self.neighbours, 'infos': self.infos}
            pkl.dump(data, f)
            print("Graph saved to {0}.".format(GRAPH_SAVE_PATH))

    def parse_orders(self):
        """ schedule the orders
        here we use multi-processing to speed up
        :return:
        """
        order_csv = pd.read_csv(ORDER_CSV_PATH)
        processes = []
        for idx in range(NUM_PROCESSES):
            start = idx * 2400 // NUM_PROCESSES
            end = (idx + 1) * 2400 // NUM_PROCESSES
            processes.append(multiprocessing.Process(target=self._parse_order_subprocess, args=(start, end)))
            processes[idx].start()
        for idx in range(NUM_PROCESSES):
            processes[idx].join()
        print("Finish parsing! Start merging the results")
        costs = []
        ratings = []
        for idx in range(NUM_PROCESSES):
            pattern = re.compile(".*rating:(\S+),.*cost:(\S+),.*")
            with open("parse_order" + str(idx), 'r') as f:
                for line in f:
                    if pattern.search(line) is None:
                        continue
                    else:
                        ratings.append(float(pattern.search(line).group(1)))
                        costs.append(float(pattern.search(line).group(2)))

        print("The average cost is {0:.2f} and the average rating is {1:.3f}".
              format(sum(costs)/len(costs), sum(ratings)/len(ratings)))
        print("{} orders can not be scheduled.".format(2400 - len(ratings)))

    def _parse_order_subprocess(self, start, end):
        order_csv = pd.read_csv(ORDER_CSV_PATH)
        with open("parse_order" + str(int(start * NUM_PROCESSES / 2400)), 'w') as f:
            for i in range(start, end):
                order = order_csv.loc[i]
                src, dest = order['CityOfSeller'], order['CityOfPurchaser']
                paths = self.find_all_paths(src, dest)
                if len(paths) == 0:
                    continue
                for path in paths:
                    strategies = self.get_path_cost(order, path)
                    best_strategy = self._find_best_strategy(strategies)
                f.write("For order {0}, found {1} strategies in total. In the best strategy:\n rating:{2:.3f}, cost:{3:.3f}, path:{4}, vehicles:{5} arrival time:{6}\n".format(
                    i, len(strategies), best_strategy['Rating'], best_strategy['Cost'], best_strategy['Path'], best_strategy['Vehicle'], best_strategy['TimeOfArrival']))

    def _find_best_strategy(self, strategies):
        best_score = 0
        best_strategy = None
        for strategy in strategies:
            score = self._evaluate(strategy['Cost'], strategy['Rating'])
            if score > best_score:
                best_score = score
                best_strategy = strategy
        return best_strategy

    @staticmethod
    def _evaluate(cost, rate):
        return rate / cost

    def get_path_cost(self, order, path):
        """ compute the cost for a specific order and path
        :param order:
        :param path:
        :return: the cost
        """
        cost_segments = [{} for _ in range(len(path)-1)]
        for i in range(len(path)-1):
            start, end = path[i], path[i+1]
            for idx, rec in enumerate(self.infos[start][end]):
                cost_segments[i][rec['Vehicle']] = self._analyze_cost_segment_path(order, rec)

        num_strategies = 1
        choices = [len(cost_segments[i]) for i in range(len(path)-1)]
        for choice in choices:
            num_strategies *= choice

        strategies = [{} for _ in range(num_strategies)]
        idx = 0
        while idx < num_strategies:
            vehicles = []
            sum_cost = 0
            time_consumed = 0
            for i, seg in enumerate(cost_segments):
                vehicle = list(seg.keys())[np.random.randint(0, choices[i])]
                vehicles.append(vehicle)
                sum_cost += seg[vehicle][0]
                time_consumed += seg[vehicle][1]
            for strategy in strategies:
                if len(strategy) != 0 and vehicles == strategy['Vehicle']:
                    break
            strategies[idx]['Vehicle'] = vehicles
            strategies[idx]['Cost'] = sum_cost
            strategies[idx]['Rating'] = self._time2rating(time_consumed, order['Emergency'])
            strategies[idx]['TimeOfArrival'] = sec2std(std2sec(order['TimeOfOrder']) + time_consumed)
            strategies[idx]['Path'] = path
            idx = idx + 1

        return strategies

    @staticmethod
    def _time2rating(time_consumed, emergency=0):
        """ return the rating according to the time consumption
        :param time_consumed: time it takes to reach the arrival city
        :param emergency: whether the order is emergency
        :return:
        """
        if emergency:
            # if emergency, we set the unit time to be half day
            unit = 12 * 60 * 60
        else:
            unit = 24 * 60 * 60

        if time_consumed < unit:
            rate = 5
        elif time_consumed < 2 * unit:
            rate = 4
        elif time_consumed < 3 * unit:
            rate = 3
        elif time_consumed < 4 * unit:
            rate = 2
        else:
            rate = 1
        return rate

    @staticmethod
    def _analyze_cost_segment_path(order, rec):
        """ given the order and rec, compute corresponding cost and consumer rate
        :param order:
        :param rec:
        :return: the tuple (cost, rate)
        """
        commodity_amount = order['AmountOfCommodity']
        order_time = std2sec(order['TimeOfOrder'])
        depart_time = std2sec(rec['TimeOfDeparture'])
        speed = rec['Speed']
        delay = rec['AverageDelay']
        unit_cost = rec['Cost']
        dist = rec['dist']

        cost = dist / 50 * commodity_amount * unit_cost
        time_consumed = delay * 60 + dist / speed * 60 * 60 + depart_time - order_time
        if order_time >= depart_time:
            # order time is later than depart time
            time_consumed += 24 * 60 * 60

        return cost, time_consumed


if __name__ == '__main__':
    graph = Graph()
    graph.parse_orders()
