#!/usr/bin/env python
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog

import os.path as osp
import time
import numpy as np
import pandas as pd
import pickle as pkl
import warnings
from tqdm import trange
from collections import deque
from src.config import cfg
from src.utils import std2min
warnings.filterwarnings('ignore')


class Hub(object):
    def __init__(self, index, neighbours):
        self.index = index
        self.neighbours = neighbours


class Info(object):
    def __init__(self, src, dest, time_on_way, unit_cost_trip, depart_time, vehicle_type):
        """
        Data structure to store the information of a specific transportation tool

        :param src: index of the city of departure
        :param dest: index of the city of arrival
        :param time_on_way: average delay + distance / speed  (unit: min)
        :param unit_cost_trip: cost it takes to transmit 1kg between the two cities
        :param depart_time: time of departure
        :param vehicle_type: e.g. ship, plane, train, truck
        """
        self.src = src
        self.dest = dest
        self.vehicle_type = vehicle_type
        self.time_on_way = time_on_way
        self.unit_cost_trip = unit_cost_trip
        self.depart_time = depart_time
        self.arrival_time = self.depart_time + self.time_on_way
        while self.arrival_time >= 1440:
            self.arrival_time -= 1440

    def get_time_cost(self, total_weight, prev_arrival, emergency=0, ratio=1):
        """
        get the time consumption and total cost between two specific cities with specific vehicle

        :param total_weight: total weight of the order
        :param prev_arrival: arrival time of previous vehicle(in the form of minutes)
        :param emergency: attribute of the order
        :param ratio: used for hubs(default: 1)
        :return:
        """
        time_consumed = self.time_on_way + self.depart_time - prev_arrival
        if prev_arrival > self.depart_time:
            time_consumed += 1440
        if not emergency:
            total_cost = cfg.PARAM.WEIGHT_AMOUNT * total_weight * ratio * self.unit_cost_trip \
                + cfg.PARAM.WEIGHT_TIME * time_consumed
        else:
            total_cost = cfg.PARAM.WEIGHT_AMOUNT * total_weight * ratio * self.unit_cost_trip \
                + cfg.PARAM.WEIGHT_TIME_EMERGENCY * time_consumed
        return time_consumed, total_cost


class Graph(object):
    def __init__(self, path, constraint=False, graph_type=1, tune_mode=False):
        """
        use the graph to model the city network

        :param path: path to store the model(in pkl format)
        :param constraint: whether we will consider some constraints(e.g. problem3)
        :param graph_type: type of the network({1: complete graph}, {2: graph only contains stations(in problem4)}, {3: G_1/G_2}
        """
        self.path = path
        self.constraint = constraint
        self.graph_type = graph_type
        self.tune_mode = tune_mode
        self.distance = np.array(pd.read_csv(cfg.DATA.DISTANCE_CSV))
        self.neighbours = [set() for _ in range(cfg.NUM_CITIES+1)]
        self.large_cities, self.small_cities = None, None
        self.infos = [[[] for i in range(cfg.NUM_CITIES+1)] for j in range(cfg.NUM_CITIES+1)]
        # read in the commodity information
        commodity_csv = pd.read_csv(cfg.DATA.COMMODITY_CSV)
        commodity_csv['CategoryOfCommodity'] = commodity_csv['CategoryOfCommodity'].map(cfg.COMMODITY_TYPES)
        self.commodity_index2types = [0 for _ in range(len(commodity_csv) + 1)]
        self.commodity_index2weights = [0 for _ in range(len(commodity_csv) + 1)]
        for i in range(len(commodity_csv)):
            commodity = commodity_csv.loc[i]
            index, commodity_type, weight = int(commodity[0]), int(commodity[2]), commodity[3]
            self.commodity_index2types[index] = commodity_type
            self.commodity_index2weights[index] = weight
        self._build_graph()

    def _build_graph(self):
        since = time.time()
        print("Start building graph...")
        if osp.exists(self.path):
            with open(self.path, "rb") as f:
                model = pkl.load(f)
            self.neighbours = model["neighbours"]
            self.infos = model["infos"]
            if "large_cities" in model:
                self.large_cities = model["large_cities"]
            if "small_cities" in model:
                self.small_cities = model["small_cities"]
            print("Load model from {} successfully.".format(self.path))
            print("Graph building finished in {:.3f}s".format(time.time() - since))
            return
        vehicle_csv = pd.read_csv(cfg.DATA.VEHICLE_CSV)
        # if we are building the graph which only contains the large stations
        if self.graph_type == 2:
            self.large_cities = vehicle_csv.loc[vehicle_csv["Vehicle"] == "Plane", "IndexOfDepartureCity"].unique().tolist()
            # filter the rows whose departure city is not a station
            vehicle_csv = vehicle_csv.loc[(vehicle_csv["IndexOfDepartureCity"].isin(self.large_cities)) &
                                          (vehicle_csv["IndexOfArrivalCity"].isin(self.large_cities))]
            # use the set for faster speed
            self.large_cities = set(self.large_cities)
        # if we are building the complement graph
        elif self.graph_type == 3:
            self.large_cities = vehicle_csv.loc[vehicle_csv["Vehicle"] == "Plane", "IndexOfDepartureCity"].unique().tolist()
            self.small_cities = [city for city in list(range(1, cfg.NUM_CITIES+1)) if city not in self.large_cities]
            vehicle_csv = vehicle_csv.loc[(vehicle_csv["IndexOfDepartureCity"].isin(self.small_cities)) &
                                          (vehicle_csv["IndexOfArrivalCity"].isin(self.small_cities))]
            self.large_cities = set(self.large_cities)
            self.small_cities = set(self.small_cities)

        # use index as after filtering, the indices may be inconsistent
        vehicle_index = vehicle_csv.index
        t = trange(len(vehicle_csv), desc="Parsing the vehicle.csv...")
        for i in t:
            t.set_description("Processing {}".format(i))
            vehicle = vehicle_csv.loc[vehicle_index[i]]
            # get index of departure city and arrival city
            src, dest = vehicle[0], vehicle[1]
            # get the average delay time and speed
            delay, speed = vehicle[2], vehicle[3]
            # get the unit cost
            unit_cost = vehicle[4]
            # get the type of the vehicle(e.g. plane)
            depart_time, vehicle_type = vehicle[5], vehicle[6]
            # calculate relevant information and store
            time_on_way = delay + self.distance[src-1, dest-1] / speed * 60
            # calculate the unit cost of the whole trip
            unit_cost_trip = self.distance[src-1, dest-1] * unit_cost / 50
            info = Info(src, dest, time_on_way, unit_cost_trip, std2min(depart_time), vehicle_type)
            # store
            self.neighbours[src].add(dest)
            self.infos[src][dest].append(info)

        model = dict()
        model["neighbours"], model["infos"] = self.neighbours, self.infos
        if self.large_cities is not None:
            model["large_cities"] = self.large_cities
        if self.small_cities is not None:
            model["small_cities"] = self.small_cities
        with open(self.path, "wb") as f:
            pkl.dump(model, f)
        print("Graph building finished in {:.3f}s. Dumped into {}".format(time.time() - since, self.path))

    def solve_single_order(self, order, all_vertices=False):
        """
        use dijkstra algorithm to solve

        :param order: order to be analyzed
        :param all_vertices: if True, single-source, all destinations(Note that this will shadow the dest)
        :return: if all_vertices=False, return tuple(infos, cost), else return dict containing several tuples
        """
        if all_vertices:
            assert self.small_cities is not None, "You can only set all_vertices=True in the small graph"
        src, dest = order[0], order[1]
        commodity_type = self.commodity_index2types[order[3]]
        commodity_weight = self.commodity_index2weights[order[3]]
        amount = order[4]
        emergency = order[5]
        total_weight = commodity_weight * amount
        # next, we run the dijkstra algorithm
        if self.graph_type == 1:
            vertices = list(range(cfg.NUM_CITIES+1))
        elif self.graph_type == 2:
            vertices = self.large_cities.copy()
        else:
            vertices = self.small_cities.copy()
        # we maintain the distance and best info for each vertex
        dist_infos = {vertex: {"dist": np.inf, "info": None} for vertex in vertices}
        previous_vertices = {vertex: None for vertex in vertices}
        dist_infos[src]["dist"] = 0
        while vertices:
            current_vertex = min(vertices, key=lambda x: dist_infos[x]["dist"])
            best_info = dist_infos[current_vertex]["info"]
            prev_arrival = best_info.arrival_time if current_vertex != src else std2min(order[2])
            if best_info is not None:
                # if we have found the destination
                if not all_vertices and best_info.dest == dest:
                    break

            if dist_infos[current_vertex]["dist"] == np.inf:
                break

            for neighbour in self.neighbours[current_vertex]:
                # then we iterate all the possible vehicles
                min_cost = cfg.INT_MAX
                best_info = None
                for info in self.infos[current_vertex][neighbour]:
                    # consider the constraint: (type of commodity, vehicle type)
                    if self.constraint and (commodity_type, cfg.VEHICLES.index(info.vehicle_type)) in cfg.CONSTRAINTS:
                        continue
                    _, info_cost = info.get_time_cost(total_weight, prev_arrival, emergency)
                    if info_cost < min_cost:
                        min_cost = info_cost
                        best_info = info
                if min_cost + dist_infos[current_vertex]["dist"] < dist_infos[neighbour]["dist"]:
                    # update
                    dist_infos[neighbour]["dist"] = min_cost + dist_infos[current_vertex]["dist"]
                    dist_infos[neighbour]["info"] = best_info
                    previous_vertices[neighbour] = current_vertex

            # remove current vertex from the vertices
            vertices.remove(current_vertex)
        # if all_vertices = False, we only need to find the shortest path between the src and dest
        if not all_vertices:
            infos, current_vertex = deque(), dest
            while previous_vertices[current_vertex] is not None:
                infos.appendleft(dist_infos[current_vertex]["info"])
                current_vertex = previous_vertices[current_vertex]
            return infos
        # if True, we will return dictionary containing information about each neighbour
        else:
            info_costs = dict()
            for dest in self.small_cities:
                if src == dest:
                    continue
                costs = []
                infos, current_vertex = deque(), dest
                while previous_vertices[current_vertex] is not None:
                    infos.appendleft(dist_infos[current_vertex]["info"])
                    costs.append(dist_infos[current_vertex]["dist"])
                    current_vertex = previous_vertices[current_vertex]
                info_costs[dest] = (infos, sum(costs))
            return info_costs

