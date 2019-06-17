#!/usr/bin/env python
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog

"""
Problem Solver
when I was writing the code, only god and I knew what I was writing about.
Now only the god knows.
"""

import os
import os.path as osp
import time
import pickle as pkl
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import trange
from src.utils import std2min, min2std
from src.graph import Graph, Info, Hub
from src.config import cfg
from src.progress.bar import Bar
import warnings
warnings.filterwarnings('ignore')


def strategy_sort_func(x, info, city, func_type=0):
    """
    function used in sorting strategies( when choosing packages for hubs in problem3)

    :param x: the strategy
    :param info: refer to the "Info" class, containing the information of edge
    :param city: index of the hub
    :param func_type: two types for knapsack(0: divide by total weight)
    :return:
    """
    total_weight, emergency = x["TotalWeight"], x["emergency"]
    city_idx = x["Paths"][:-1].index(city)
    prev_arrival = x["Infos"][city_idx].arrival_time if city_idx != 0 else x["OrderTime"]
    prev_cost = x["Costs"][city_idx]
    cur_cost = info.get_time_cost(total_weight, prev_arrival, emergency, cfg.PARAM.HUB_UNIT_COST_RATIO)[1]
    if city_idx < len(x["Paths"]) - 2:
        cur_cost += x["Infos"][city_idx + 1].get_time_cost(total_weight, info.arrival_time, emergency)[1]
        prev_cost += x["Costs"][city_idx + 1]
    if func_type == 0:
        return (prev_cost - cur_cost) / total_weight
    return prev_cost - cur_cost


class Solver(object):
    def __init__(self):
        # default value
        self.problem_id = 2
        self.base_dir = cfg.PROB.PROBLEM2_PATH
        self.hub_path = osp.join(cfg.PROB.PROBLEM2_PATH, "hubs.pkl")
        self.src_strategy_path = osp.join(cfg.PROB.PROBLEM1_PATH, "strategies.pkl")
        self.target_strategy_path = osp.join(self.base_dir, "strategies.pkl")
        self.constraint = False
        self.tune_mode = False
        # read in the distance information
        self.hubs = []
        # orders allowed to be transported by the hub, used in problem3
        self.order_allowed = None
        # neighbours allowed transportation by the hub
        self.neighbour_allowed = {}
        # large cities(cities with airline service)
        self.graph = None
        self.station_graph, self.small_graph = None, None
        self._load_data()

    def solve(self, problem_id, tune_mode=False):
        """
        solve interface

        :param problem_id: id of the problem
        :param tune_mode: whether we are tuning the parameter a and b(if True, we filter the records of emergency)
        :return:
        """
        assert problem_id in [1, 2, 3, 4], "You can only choose problem(1-4) to solve."
        self.problem_id = problem_id
        self.tune_mode = tune_mode

        self.constraint = True if problem_id == 3 else False

        self.base_dir = cfg.PROB.PROBLEM_LIST[problem_id-1]
        self.hub_path = osp.join(cfg.PROB.PROBLEM2_PATH if problem_id == 2 else cfg.PROB.PROBLEM3_PATH, "hubs.pkl")
        self.src_strategy_path = osp.join(cfg.PROB.PROBLEM1_PATH if problem_id == 2 else cfg.PROB.PROBLEM3_PATH, "strategies.pkl")
        self.target_strategy_path = osp.join(self.base_dir, "strategies.pkl")

        if problem_id != 2:
            self._solve_order()
        if problem_id in [2, 3]:
            self._build_hubs()
            self._modify_strategies()

    def _modify_strategies(self):
        """
        for prob2 and prob3, after building hubs, we need to modify some strategies
        """
        since = time.time()
        print("Modification starts...")
        if self.tune_mode and osp.exists(self.target_strategy_path) and self.problem_id == 2:
            os.remove(self.target_strategy_path)
            print("Clean {}".format(self.target_strategy_path))

        hub_indices = [hub.index for hub in self.hubs]
        assert len(hub_indices) > 0, "No hub there!"
        with open(self.src_strategy_path, "rb") as f:
            strategies = pkl.load(f)
        for idx, strategy in enumerate(strategies):
            paths = strategy["Paths"]
            # old_time_consumption = strategy["TimeConsumption"]
            if self.constraint and not self.order_allowed[strategy["index"]]:
                continue
            for i in range(len(paths)-1):
                # irrelevant with the hubs
                start, end = paths[i], paths[i+1]
                # In problem3, if end city does not support our hubs
                if start not in hub_indices or self.constraint and end not in self.neighbour_allowed[start]:
                    continue
                hub = self.hubs[hub_indices.index(start)]
                # Note that hub.neighbours[end] has type "Info"
                info = hub.neighbours[end]
                try:
                    strategy['Vehicles'][i] = info.vehicle_type
                except:
                    print("Error")
                    break
                # update the cost list
                strategy["AmountCost"] += (info.unit_cost_trip * cfg.PARAM.HUB_UNIT_COST_RATIO \
                    - strategy["Infos"][i].unit_cost_trip) * strategy["TotalWeight"]
                prev_arrival = strategy["Infos"][i].arrival_time if i != 0 else strategy["OrderTime"]
                time_, strategy["Costs"][i] = info.get_time_cost(strategy["TotalWeight"], prev_arrival, strategy["emergency"], cfg.PARAM.HUB_UNIT_COST_RATIO)
                strategy["TimeConsumption"][i] = time_
                if i < len(paths) - 2:
                    time_, strategy["Costs"][i+1] = strategy["Infos"][i+1].get_time_cost(strategy["TotalWeight"],
                        info.arrival_time, strategy["emergency"])
                    strategy["TimeConsumption"][i+1] = time_
            """
            arrival_time = strategy['ArrivalTime']
            day_time = 0
            # if more than one day(e.g. (+1day)21:30:24
            if arrival_time.find('+') != -1:
                day_time = int(arrival_time[arrival_time.index('+')+1:arrival_time.index('d')]) * 24 * 60
                strategy['ArrivalTime'] = arrival_time[arrival_time.find(":")+1:]
            strategy['ArrivalTime'] = min2std(std2min(strategy['ArrivalTime'])
                - sum(old_time_consumption) + sum(strategy["TimeConsumption"]) + day_time, get_day=True)
            """
            strategies[idx] = strategy
        print("Modification finished in {:.3f}".format(time.time()-since))
        with open(self.target_strategy_path, "wb") as f:
            pkl.dump(strategies, f)
        print("Dumped to {}.".format(self.target_strategy_path))
        # also output to txt files
        self._pkl2txt(self.target_strategy_path)

    def _build_hubs(self):
        """
        Here we are only using part of the orders(i.e. orders processed by prob1)
        as the guidance to build the hubs
        First we parse the output of prob1 to get the costs for each city
        Second for each city whose sum_cost is not zero, we iterate through all the possibilities of vehicles
        to find the maximum benefit
        Third if the maximum benefit exceeds the cost of building a hub, we build a hub at that specific city
        Note that we must decide the specific vehicle for the hub and every other city it connects to
        """
        # this ensures that we must have called solve_order()
        assert osp.exists(self.src_strategy_path), \
            "Can not find {}. Perhaps you should first solve problem1".format(self.src_strategy_path)

        since = time.time()
        print("Start building the hubs...")
        # if we in the tune(check) mode, remove previous results
        if osp.exists(self.hub_path):
            if self.tune_mode:
                print("Clean {}".format(self.hub_path))
                os.remove(self.hub_path)
            else:
                with open(self.hub_path, "rb") as f:
                    model = pkl.load(f)
                    if "order_allowed" in model:
                        self.hubs = model["hubs"]
                        self.order_allowed = model["order_allowed"]
                        self.neighbour_allowed = model["neighbour_allowed"]
                    else:
                        self.hubs = model
                print("Load model from {} successfully.".format(osp.join(self.hub_path)))
                print("Hubs building finished in {:.3f}s.".format(time.time()-since))
                return

        city_costs = np.zeros((cfg.NUM_CITIES+1,), dtype=np.uint32)

        # load the strategies
        with open(self.src_strategy_path, "rb") as f:
            strategies = pkl.load(f)
        for strategy in strategies:
            # src or intermediate cities
            cities = strategy["Paths"][:-1]
            for idx, city in enumerate(cities):
                city_costs[city] += strategy["Costs"][idx]
        # note that although we takes small part of orders, only several cities will have cost = 0
        city_indices = np.argsort(-city_costs)
        t = trange(len(city_indices))
        self.order_allowed = [0 for _ in range(len(strategies))]
        for i in t:
            t.set_description("Processing {}".format(i))
            city = city_indices[i]
            if city_costs[city] == 0 or city_costs[city] < cfg.PARAM.HUB_BUILT_COST_CONST:
                continue
            self.neighbour_allowed[city] = []
            max_total_benefit = 0
            expected_total_cost, expected_total_benefit = 0, 0
            hub_neighbours = {}
            # used for hub capacity
            sum_weight = 0
            for neighbour in self.graph.neighbours[city]:
                # if the limitation is already reached
                if self.constraint and sum_weight > cfg.PARAM.HUB_CAPACITY:
                    break
                best_info = None
                # cost used for prob1,2 benefit used for problem3
                max_benefit, chosen_weight = cfg.INT_MIN, 0
                for info in self.graph.infos[city][neighbour]:
                    expected_benefit = 0
                    # count the number of allowed orders
                    allowed_count, weight = 0, 0
                    if not self.constraint:
                        cleaned_strategies = [strategy for strategy in strategies if city in strategy["Paths"][:-1]]
                        for strategy in cleaned_strategies:
                            city_idx = strategy["Paths"].index(city)
                            prev_arrival = strategy["Infos"][city_idx].arrival_time if city_idx != 0 else strategy["OrderTime"]
                            cost_ = info.get_time_cost(strategy['TotalWeight'], prev_arrival, strategy['emergency'], cfg.PARAM.HUB_UNIT_COST_RATIO)[1]
                            expected_benefit += (strategy["Costs"][city_idx] - cost_)
                            if city_idx < len(strategy["Paths"]) - 2:
                                expected_benefit += (strategy["Costs"][city_idx+1]
                                    - strategy["Infos"][city_idx+1].get_time_cost(strategy["TotalWeight"], info.arrival_time, strategy["emergency"])[1])
                    else:
                        # note that here we should consider some constraints when facing problem3
                        # e.g. some hubs may be capacitied, glass and inflammable products will be rejected
                        cleaned_strategies = [strategy for strategy in strategies if city in strategy["Paths"][:-1]
                                              and (strategy["CommodityType"], cfg.VEHICLES.index(info.vehicle_type)) not in cfg.CONSTRAINTS
                                              and strategy["CommodityType"] not in cfg.HUB_REJECTS]
                        cleaned_indices = [strategy["index"] for strategy in cleaned_strategies]
                        # then we use approximation algorithm for the knapsack problem to schedule the orders
                        # refer to slides for more details
                        # orders which are allowed to be transported by the hubs
                        # first sort the strategies
                        sorted_strategies_1 = sorted(cleaned_strategies,
                            key=lambda x: strategy_sort_func(x, info, city, func_type=0), reverse=True)
                        sorted_strategies_2 = sorted(cleaned_strategies,
                            key=lambda x: strategy_sort_func(x, info, city, func_type=1), reverse=True)

                        while len(cleaned_indices) > 0:
                            while sorted_strategies_1[0]["index"] not in cleaned_indices:
                                sorted_strategies_1.pop(0)
                            while sorted_strategies_2[0]["index"] not in cleaned_indices:
                                sorted_strategies_2.pop(0)
                            # we then chosen the better one from the two sorted strategies
                            cand1, cand2 = sorted_strategies_1[0], sorted_strategies_2[0]
                            cand1_idx, cand2_idx = cand1["Paths"][:-1].index(city), cand2["Paths"][:-1].index(city)
                            prev_arrival = cand1["Infos"][cand1_idx].arrival_time if cand1_idx != 0 else cand1["OrderTime"]
                            benefit_1 = cand1["Costs"][cand1["Paths"][:-1].index(city)] - \
                                info.get_time_cost(cand1['TotalWeight'], prev_arrival, cand1['emergency'], cfg.PARAM.HUB_UNIT_COST_RATIO)[1]
                            if cand1_idx < len(cand1["Paths"]) - 2:
                                benefit_1 += (cand1["Costs"][cand1_idx+1] - cand1["Infos"][cand1_idx+1].get_time_cost(cand1["TotalWeight"],
                                    info.arrival_time, cand1["emergency"])[1])

                            prev_arrival = cand2["Infos"][cand2_idx].arrival_time if cand2_idx != 0 else cand2["OrderTime"]
                            benefit_2 = cand2["Costs"][cand2["Paths"][:-1].index(city)] - \
                                info.get_time_cost(cand2['TotalWeight'], prev_arrival, cand2['emergency'], cfg.PARAM.HUB_UNIT_COST_RATIO)[1]
                            if cand2_idx < len(cand2["Paths"]) - 2:
                                benefit_2 += (cand2["Costs"][cand2_idx+1] - cand2["Infos"][cand2_idx+1].get_time_cost(cand2["TotalWeight"],
                                    info.arrival_time, cand2["emergency"])[1])

                            if benefit_1 < benefit_2:
                                chosen_strategy = sorted_strategies_2[0]
                                expected_benefit += benefit_2
                            else:
                                chosen_strategy = sorted_strategies_1[0]
                                expected_benefit += benefit_1
                            cleaned_indices.remove(chosen_strategy["index"])
                            if sum_weight + weight + chosen_strategy["TotalWeight"] > cfg.PARAM.HUB_CAPACITY:
                                break
                            weight += chosen_strategy["TotalWeight"]
                            allowed_count += 1
                            self.order_allowed[chosen_strategy["index"]] = 1
                            if neighbour not in self.neighbour_allowed[city]:
                                self.neighbour_allowed[city].append(neighbour)

                    # minus the variable part of hub cost(e.g. maintaining the hub)
                    if not self.constraint:
                        expected_benefit -= len(cleaned_strategies) * cfg.PARAM.HUB_BUILT_COST_VARY
                    else:
                        expected_benefit -= allowed_count * cfg.PARAM.HUB_BUILT_COST_VARY

                    if expected_benefit > max_benefit:
                        if self.constraint:
                            chosen_weight = weight
                        max_benefit = expected_benefit
                        best_info = info
                # then for all orders between the "city" and "neighbour", we all use vehicle in info
                expected_total_benefit += max_benefit
                sum_weight += chosen_weight
                hub_neighbours[neighbour] = best_info
                # print(expected_total_benefit)
                if expected_total_benefit > max_total_benefit:
                    max_total_benefit = expected_total_benefit
            if max_total_benefit - cfg.PARAM.HUB_BUILT_COST_CONST > 0:
                print("Build a hub at city {}".format(city))
                # note that the changes of vehicles will take into effective
                self.hubs.append(Hub(city, hub_neighbours))
        with open(self.hub_path, "wb") as f:
            if not self.constraint:
                pkl.dump(self.hubs, f)
            else:
                pkl.dump({"hubs": self.hubs, "neighbour_allowed": self.neighbour_allowed, "order_allowed": self.order_allowed}, f)
        # store the parsed results of orders
        print("In total we build {} hubs, saved => {}".format(len(self.hubs), self.hub_path))
        print("Hubs building finished in {:.3f}s.".format(time.time()-since))
        return

    def _solve_order(self):
        """
        use multiprocess to solve the orders
        """
        if osp.exists(self.target_strategy_path):
            print("Find strategy {}".format(osp.join(self.target_strategy_path)))
            # when we are tuning parameters, always cleaning the directory
            if not self.tune_mode:
                key = input("Clean or not?(y or n)")
            else:
                key = 'Y'
            if key == 'Y' or key == 'y':
                os.remove(self.target_strategy_path)
                os.remove(self.target_strategy_path[:-3] + "txt")
                print("Clean finished!")
            else:
                return

        if self.problem_id == 4:
            # build the station graph, note that the complete graph was built in initialization part
            self.station_graph = Graph(cfg.MODEL.GRAPH_STATION_PATH, graph_type=2)
            # build the small graph(namely complete graph/station graph
            self.small_graph = Graph(cfg.MODEL.GRAPH_SMALL_PATH, graph_type=3)

        processes = []
        for idx in range(cfg.PARAM.NUM_PROCESSES):
            start = idx * cfg.PARAM.NUM_ORDERS // cfg.PARAM.NUM_PROCESSES
            end = (idx + 1) * cfg.PARAM.NUM_ORDERS // cfg.PARAM.NUM_PROCESSES
            if self.problem_id == 4:
                processes.append(multiprocessing.Process(target=self._solve_order_station_subprocess, args=(start, end)))
            else:
                processes.append(multiprocessing.Process(target=self._solve_order_subprocess, args=(start, end)))
            processes[idx].start()
        for idx in range(cfg.PARAM.NUM_PROCESSES):
            processes[idx].join()
        print("Finish parsing! Start merging the results")
        # merge the pickle files
        strategies = []
        assert len(os.listdir(self.base_dir)) > 0, "Error! Empty directory {}".format(self.base_dir)
        for idx in range(cfg.PARAM.NUM_PROCESSES):
            filename = "parse_order" + str(idx) + ".pkl"
            with open(osp.join(self.base_dir, filename), "rb") as f:
                strategies.extend(pkl.load(f))
            # remove intermediate results
            os.remove(osp.join(self.base_dir, filename))
        with open(self.target_strategy_path, "wb") as f:
            pkl.dump(strategies, f)
        print("Finished! Merged into {}".format(osp.join(self.target_strategy_path)))
        # also output to the txt file
        self._pkl2txt(self.target_strategy_path)

    def _solve_order_subprocess(self, start, end):
        since = time.time()
        # where we stand
        idx = int(start * cfg.PARAM.NUM_PROCESSES) // cfg.PARAM.NUM_ORDERS
        order_csv = pd.read_csv(cfg.DATA.ORDER_CSV)
        if self.tune_mode:
            order_csv = order_csv.loc[order_csv["Emergency"] == 0]
        indices = order_csv.index
        # f = open(osp.join(OUTPUT_DIR, "parse_order"+str(int(idx))), "w")
        # for user-friendly output
        if idx == 0:
            print("Note that your are using {} processes and here we only show output of process 1.".format(cfg.PARAM.NUM_PROCESSES))
            bar = Bar("Parsing the orders", max=(end-start))
        strategies = []

        for i in range(start, end):
            once_since = time.time()
            order = order_csv.loc[indices[i]]
            infos = self.graph.solve_single_order(order)
            # analyze to get the strategy
            strategy = self._analyze_strategy(i, order, infos)
            strategies.append(strategy)
            if idx == 0:
                bar.suffix = "Processing {}/{} | Time: {:.3f}s | Total Time: {:.0f}m {:.0f}s".format(
                    i, end-start, time.time()-once_since, int(time.time()-since)//60, int(time.time()-since) % 60)
                bar.next()
        time_elapsed = time.time() - since
        # for process1: finish the progress bar
        if idx == 0:
            bar.finish()
            print("Finish processing {} orders in {:.0f}m {:.0f}s.".format(end - start, int(time_elapsed) // 60, int(time_elapsed) % 60))

        with open(osp.join(self.base_dir, "parse_order"+str(idx))+".pkl", "wb") as f:
            pkl.dump(strategies, f)

    def _solve_order_station_subprocess(self, start, end):
        """
        Algorithm to solve problem4:
        First we construct the station_graph(i.e. graph which only contains cities with airline service)
        Second we construct the small_graph(i.e. the complement of station_graph)
        Then we schedule the orders: there are 4 cases in total
              1.  src, dest are both stations: Run dijkstra on the station_graph
              2.  src is station and dest is not: Run dijkstra on small graph to find all shortest paths from dest.
                  Then find the station t with shortest distance to dest. Finally run dijkstra to find shortest path
                  between src and t
              3.  src is not station but dest is: Similar to case 2
              4.  src and dest are neither station: go through the steps in case2 for both src and dest, then combine
        Refer to our report for more details

        :param start: first index of the order
        :param end:  last index of the order
        """
        since = time.time()
        idx = int(start * cfg.PARAM.NUM_PROCESSES) // cfg.PARAM.NUM_ORDERS
        strategies = []
        if idx == 0:
            print("Note that you are using {} processes and here we only show the output of process1.".format(cfg.PARAM.NUM_PROCESSES))
            bar = Bar("Processing orders", max=end-start)
        for i in range(start, end):
            once_since = time.time()
            order = pd.read_csv(cfg.DATA.ORDER_CSV).loc[i]
            # copy the order
            order_v = order.copy()
            src, dest = order[0], order[1]
            total_weight, emergency = self.commodity_index2weights[order[3]] * order[4], order[5]
            substation_s, substation_t = src, dest
            src_s_infos, t_dest_infos = None, None
            infos = []
            if src not in self.station_graph.large_cities:
                # if src is a small city, we run dijkstra on small_graph to get substation_s
                info_costs = self.small_graph.solve_single_order(order_v, all_vertices=True)
                min_cost = cfg.INT_MAX
                best_info_station, best_info_small = None, None
                for vertex, (infos_small, cost) in info_costs.items():
                    # then we find station which can be directly reached by the vertex in the complete graph
                    for neighbour in self.graph.neighbours[vertex]:
                        # if neighbour is not a station, continue
                        if neighbour not in self.station_graph.large_cities:
                            continue
                        prev_arrival = infos_small[-1].arrival_time
                        for info in self.graph.infos[vertex][neighbour]:
                            _, cost_ = info.get_time_cost(total_weight, prev_arrival, emergency)
                            if cost + cost_ < min_cost:
                                min_cost = cost + cost_
                                best_info_station = info
                                best_info_small = infos_small
                src_s_infos = list(best_info_small) + [best_info_station]
                # now we get the substation
                substation_s = src_s_infos[-1].dest
            if dest not in self.station_graph.large_cities:
                # if dest is a small city, we run dijkstra on small_graph to get substation_t
                # just follow the above steps
                order_v[0] = dest
                info_costs = self.small_graph.solve_single_order(order_v, all_vertices=True)
                min_cost = cfg.INT_MAX
                best_info_small_rev, best_info_station_rev = None, None
                for vertex, (infos_small, cost) in info_costs.items():
                    # then we find station which can be directly reached by the vertex in the complete graph
                    for neighbour in self.graph.neighbours[vertex]:
                        # if neighbour is not a station, continue
                        if neighbour not in self.small_graph.large_cities or vertex not in self.graph.neighbours[neighbour]:
                            continue
                        prev_arrival = std2min(order[2])
                        for info in self.graph.infos[vertex][neighbour]:
                            _, cost_ = info.get_time_cost(total_weight, prev_arrival, emergency)
                            if cost + cost_ < min_cost:
                                min_cost = cost + cost_
                                best_info_small_rev = infos_small
                                best_info_station_rev = info
                # now we get the substation
                substation_t = best_info_station_rev.dest
                order_v[0], order_v[1] = best_info_small_rev[-1].dest, dest
                best_info_small = self.small_graph.solve_single_order(order_v)
                best_info_station = None
                min_cost = cfg.INT_MAX
                prev_arrival = std2min(order[2])
                for info in self.graph.infos[substation_t][order_v[0]]:
                    _, cost_ = info.get_time_cost(total_weight, prev_arrival, emergency)
                    if cost_ < min_cost:
                        min_cost = cost_
                        best_info_station = info
                t_dest_infos = [best_info_station] + list(best_info_small)

            # then find shortest path on small graph between substation_s and substation_t
            order_v[0], order_v[1] = substation_s, substation_t
            s_t_infos = self.station_graph.solve_single_order(order_v)
            # combine the three parts
            if src_s_infos is not None:
                infos.extend(src_s_infos)
            infos.extend(list(s_t_infos))
            if t_dest_infos is not None:
                infos.extend(t_dest_infos)
            # then give the strategy
            strategy = self._analyze_strategy(i, order, infos)
            strategies.append(strategy)
            if idx == 0:
                bar.suffix = "Processing {}/{} | Time: {:.3f}s | Total Time: {:.0f}m {:.0f}s".format(
                    i, end-start, time.time()-once_since, int(time.time()-since)//60, int(time.time()-since) % 60)
                bar.next()
        time_elapsed = time.time() - since
        if idx == 0:
            bar.finish()
            print("Finish processing {} orders in {:.0f}m {:.0f}s.".format(end - start, int(time_elapsed) // 60, int(time_elapsed) % 60))

        with open(osp.join(self.base_dir, "parse_order"+str(idx))+".pkl", "wb") as f:
            pkl.dump(strategies, f)

    def _analyze_strategy(self, idx, order, infos):
        """
        analyze the information of edges and return the strategy

        :param idx: index of the order
        :param order: order
        :param infos: information of the edges on the optimal path(type: list)
        :return: strategy(type: dict)
        """
        order_time = order[2]
        amount, emergency = order[4], order[5]
        paths, vehicles, costs, time_consumed = [], [], [], []
        total_weight = amount * self.commodity_index2weights[order[3]]
        prev_arrival = std2min(order_time)
        amount_cost = 0
        for info in infos:
            amount_cost += total_weight * info.unit_cost_trip
            time_, cost_ = info.get_time_cost(total_weight, prev_arrival, emergency)
            costs.append(cost_)
            time_consumed.append(time_)
            prev_arrival = info.arrival_time
            paths.append(info.src)
            vehicles.append(info.vehicle_type)
        paths.append(infos[-1].dest)
        strategy = dict()
        # add commodity type as we need it in problem3
        # add index as we will use later in the sorting
        strategy["index"], strategy["emergency"] = idx, emergency
        strategy["CommodityType"], strategy["TotalWeight"] = self.commodity_index2types[order[3]], total_weight
        strategy["Paths"], strategy["Costs"], strategy["TimeConsumption"] = paths, costs, time_consumed
        strategy["Vehicles"], strategy["AmountCost"] = vehicles, amount_cost
        # very very very very don't want to add the infos
        # but I really really don't want to reorganize the code
        strategy["Infos"], strategy["OrderTime"] = infos, std2min(order_time)
        return strategy

    def _load_data(self):
        """
        some preparation work
        """
        if not osp.exists(cfg.MODEL.ROOT_DIR):
            os.makedirs(cfg.MODEL.ROOT_DIR)
        for output_dir in cfg.PROB.PROBLEM_LIST:
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
        # read in the commodity information
        commodity_csv = pd.read_csv(cfg.DATA.COMMODITY_CSV)
        commodity_csv['CategoryOfCommodity'] = commodity_csv['CategoryOfCommodity'].map(cfg.COMMODITY_TYPES)
        self.commodity_index2types = [0 for _ in range(len(commodity_csv)+1)]
        self.commodity_index2weights = [0 for _ in range(len(commodity_csv)+1)]
        for i in range(len(commodity_csv)):
            commodity = commodity_csv.loc[i]
            index, commodity_type, weight = int(commodity[0]), int(commodity[2]), commodity[3]
            self.commodity_index2types[index] = commodity_type
            self.commodity_index2weights[index] = weight
        self.graph = Graph(cfg.MODEL.GRAPH_PATH)

    @staticmethod
    def _pkl2txt(path):
        with open(path , "rb") as f:
            strategies = pkl.load(f)
        txt_path = osp.join(osp.dirname(path), osp.splitext(osp.basename(path))[0] + ".txt")
        with open(txt_path, "w") as f:
            for idx, strategy in enumerate(strategies):
                f.write("For order {} with totalWeight: {:.3f} and emergency: {}, Best strategy is:\n".format(
                    idx, strategy["TotalWeight"], strategy["emergency"]))
                total_time = sum(strategy["TimeConsumption"])
                arrival_time = min2std(strategy["OrderTime"]+total_time, get_day=True)
                f.write("Paths: {}, Vehicles: {}, AmountCost: {:.3f}$, TimeCost: {:.2f}m, ArrivalTime: {}\n"
                   .format(strategy["Paths"], strategy["Vehicles"], strategy["AmountCost"], total_time, arrival_time))
                """
                # more detailed information
                for info in strategy["Infos"]:
                    f.write("From {} to {}, taking {}, departure time:{}, arrival time:{}\n".format(
                        info.src, info.dest, info.vehicle_type, min2std(info.depart_time), min2std(info.arrival_time)))
                f.write('\n')
                """
        print("Txt file saved to {}".format(txt_path))
