#!/usr/bin/env python
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog

import os
import os.path as osp
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from easydict import EasyDict as edict
from src.config import cfg, merge_a_into_b
from src.solver import Solver

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["figure.figsize"] = (12, 8)


class Evaluator(object):
    def __init__(self):
        self.prob_id, self.base_dir = None, None
        self.strategy_path, self.csv_path = None, None
        self.hub_path = None

    def evaluate(self, prob_id, tune=False, plot=True):
        assert prob_id in [1, 2, 3, 4], "Only support evaluation for problems1~4"

        self.prob_id = prob_id
        self.base_dir = osp.join(cfg.EVALUATION.ROOT_DIR, "prob" + str(prob_id))
        if not osp.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.strategy_path = osp.join(cfg.PROB.PROBLEM_LIST[prob_id-1], "strategies.pkl")
        self.csv_path = self.strategy_path[:-3] + "csv"
        self.hub_path = osp.join(osp.dirname(self.strategy_path), "hubs.pkl")
        if not tune and not osp.exists(self.strategy_path):
            key = input("It seems that you have not run solver for this problem yet. Run Now?(y or n)")
            if key == 'Y' or key == 'y':
                solver = Solver()
                solver.solve(problem_id=prob_id)
            else:
                print("exit")
                return

        if osp.exists(self.strategy_path):
            self._pkl2csv()
            avg_amount_cost, avg_time_cost = self.get_avg_cost()
            print("For problem {}:\nAverage amount cost:{:.2f} | Average time cost: {:.2f}min".
                format(prob_id, avg_amount_cost, avg_time_cost))

        # whether we are tuning the parameters
        if tune:
            if prob_id == 1:
                # for problem1 we tune the weight_amount and weight_cost
                # self.weight_tune()
                self.weight_plot()
            elif prob_id == 2:
                self.hub_cost_test()
                self.hub_cost_plot()
            elif prob_id == 3:
                self.hub_cap_test()
                self.hub_cap_plot()

        # plot the distribution or not
        if plot:
            self.plot_distribution()

    def hub_cost_test(self):
        """ sensitivity check for parameters cfg.PARAM.HUB_BUILT_COST_VARY
        cfg.PARAM.HUB_BUILT_COST_CONST, cfg.PARAM.HUB_BUILT_COST_RATIO
        """
        cost_vary_path = osp.join(self.base_dir, "vary.csv")
        cost_const_path = osp.join(self.base_dir, "const.csv")
        cost_ratio_path = osp.join(self.base_dir, "ratio.csv")

        new_dict = edict()
        ori_ratio, ori_const_cost, ori_vary_cost = cfg.PARAM.HUB_UNIT_COST_RATIO, \
            cfg.PARAM.HUB_BUILT_COST_CONST, cfg.PARAM.HUB_BUILT_COST_VARY

        num_hubs_list, amount_cost_list, time_cost_list = [], [], []
        # insert some values around the default values
        index = np.where(cfg.EVALUATION.HUB_RATIO_FIELD == ori_ratio)[0]
        hub_ratio_field = np.insert(cfg.EVALUATION.HUB_RATIO_FIELD, index, round(0.95 * ori_ratio, 2))
        hub_ratio_field = np.insert(hub_ratio_field, index+2, round(1.05 * ori_ratio, 2))

        for ratio in hub_ratio_field:
            new_dict["HUB_UNIT_COST_RATIO"] = ratio
            merge_a_into_b(new_dict, cfg)
            print(cfg)
            solver = Solver()
            solver.solve(problem_id=2, tune_mode=True)
            with open(self.hub_path, "rb") as f:
                num_hubs = len(pkl.load(f))
            if num_hubs == 0:
                print("When ratio is: {}. Some error has occurred... Maybe no hubs were built.".format(ratio))
                continue
            self._pkl2csv()
            # get the number of hubs
            avg_amount_cost, avg_time_cost = self.get_avg_cost()
            amount_cost_list.append(avg_amount_cost)
            time_cost_list.append(avg_time_cost)
            num_hubs_list.append(num_hubs)
            print("Ratio: {:.2f} | Hubs: {:.0f} | Average amount cost:{:.2f}$ | Average time cost:{:.2f}m.".
                  format(ratio, num_hubs, avg_amount_cost, avg_time_cost))
        # save to csv
        df = pd.DataFrame(data=list(zip(cfg.EVALUATION.HUB_RATIO_FIELD, num_hubs_list, amount_cost_list, time_cost_list)),
            columns=["Ratio", "NumHubs", "AmountCost", "TimeCost"])
        df["Products"] = df["AmountCost"] * df["TimeCost"]
        df["Value"] = cfg.PARAM.WEIGHT_AMOUNT * df["AmountCost"] + cfg.PARAM.WEIGHT_TIME * df["TimeCost"]
        df.to_csv(cost_ratio_path, index=False, float_format="%.2f")
        
        # reset to default value
        new_dict["HUB_UNIT_COST_RATIO"] = ori_ratio
        index = np.where(cfg.EVALUATION.HUB_COST_CONST_FIELD == ori_const_cost)[0]
        hub_const_field = np.insert(cfg.EVALUATION.HUB_COST_CONST_FIELD, index, 0.95 * ori_const_cost)
        hub_const_field = np.insert(hub_const_field, index+2, 1.05 * ori_const_cost)
        print(cfg.EVALUATION.HUB_COST_CONST_FIELD)

        # then check the sensitivity of the HUB_BUILT_COST_CONST
        # add exception handler for fear of no hubs
        num_hubs_list, amount_cost_list, time_cost_list =[], [], []
        for const_cost in hub_const_field:
            new_dict["HUB_BUILT_COST_CONST"] = const_cost
            merge_a_into_b(new_dict, cfg)
            print(cfg)
            solver = Solver()
            solver.solve(problem_id=2, tune_mode=True)
            with open(self.hub_path, "rb") as f:
                num_hubs = len(pkl.load(f))
            if num_hubs == 0:
                print("When const cost is :{}. No hubs are built. Continue...".format(const_cost))
                continue
            self._pkl2csv()
            # get the number of hubs
            avg_amount_cost, avg_time_cost = self.get_avg_cost()
            amount_cost_list.append(avg_amount_cost)
            time_cost_list.append(avg_time_cost)
            num_hubs_list.append(num_hubs)
            print("Const cost: {:.0f} | Hubs: {:.0f} | Average amount cost: {:.2f}$ | Average time cost: {:.2f}m.".
                  format(const_cost, num_hubs, avg_amount_cost, avg_time_cost))
        df = pd.DataFrame(data=list(zip(cfg.EVALUATION.HUB_COST_CONST_FIELD, num_hubs_list, amount_cost_list, time_cost_list)),
                          columns=["HubConstCost", "NumHubs", "AmountCost", "TimeCost"])
        df["Products"] = df["AmountCost"] * df["TimeCost"]
        df["Value"] = cfg.PARAM.WEIGHT_AMOUNT * df["AmountCost"] + cfg.PARAM.WEIGHT_TIME * df["TimeCost"]
        df.to_csv(cost_const_path, index=False, float_format="%.2f")

        # reset the const part to default value
        new_dict["HUB_BUILT_COST_CONST"] = ori_const_cost

        index = np.where(cfg.EVALUATION.HUB_COST_VARY_FIELD == ori_vary_cost)[0]
        hub_vary_field = np.insert(cfg.EVALUATION.HUB_COST_VARY_FIELD, index, 0.95 * ori_vary_cost)
        cfg.EVALUATION.HUB_COST_VARY_FIELD = np.insert(hub_vary_field, index+2, 1.05 * ori_vary_cost)

        # check the sensitivity of the HUB_BUILT_COST_VARY
        num_hubs_list, amount_cost_list, time_cost_list = [], [], []
        for vary_cost in hub_vary_field:
            new_dict["HUB_BUILT_COST_VARY"] = vary_cost
            merge_a_into_b(new_dict, cfg)
            print(cfg)
            solver = Solver()
            solver.solve(problem_id=2, tune_mode=True)
            with open(self.hub_path, "rb") as f:
                num_hubs = len(pkl.load(f))
            if num_hubs == 0:
                print("When vary cost is :{}. No hubs are built. Continue..".format(vary_cost))
                continue
            self._pkl2csv()
            avg_amount_cost, avg_time_cost = self.get_avg_cost()
            amount_cost_list.append(avg_amount_cost)
            time_cost_list.append(avg_time_cost)
            num_hubs_list.append(num_hubs)
            print("Vary cost: {:.0f} | Hubs: {:.0f} | Average amount cost: {:.3f}$ | Average time cost: {:.2f}m.".
                  format(vary_cost, num_hubs, avg_amount_cost, avg_time_cost))
        df = pd.DataFrame(data=list(zip(cfg.EVALUATION.HUB_COST_VARY_FIELD, num_hubs_list, amount_cost_list, time_cost_list)),
                          columns=["HubVaryCost", "NumHubs", "AmountCost", "TimeCost"])
        df["Products"] = df["AmountCost"] * df["TimeCost"]
        df["Value"] = cfg.PARAM.WEIGHT_AMOUNT * df["AmountCost"] + cfg.PARAM.WEIGHT_TIME * df["TimeCost"]
        df.to_csv(cost_vary_path, index=False, float_format="%.2f")
        # reset
        new_dict["HUB_BUILT_COST_VARY"] = ori_vary_cost
        merge_a_into_b(new_dict, cfg)

    def hub_cost_plot(self):
        cost_vary_path = osp.join(self.base_dir, "vary.csv")
        cost_const_path = osp.join(self.base_dir, "const.csv")
        cost_ratio_path = osp.join(self.base_dir, "ratio.csv")

        # plot the ratio
        plt.figure()
        data = pd.read_csv(cost_ratio_path)
        x, y = list(data["Ratio"]), list(data["Values"])
        plt.plot(x, y, marker="o", ms=8)
        index = data.loc[data["Ratio"] == cfg.PARAM.HUB_UNIT_COST_RATIO].index[0]
        x1, y1 = data.loc[index-1, ["Ratio", "Values"]]
        x2, y2 = data.loc[index+1, ["Ratio", "Values"]]

        # calculate the slope
        slope = (y2 - y1) / (x2 - x1)
        new_x = np.linspace(0.9*min(x), 1.1*max(x), 20)
        new_y = slope * (new_x - x1) + y1
        plt.plot(new_x, new_y, '--', linewidth=2)
        plt.xlabel("ratio")
        plt.ylabel("value")
        plt.title("Objective(ax + by)-HubRatio Curve\n" + "Slope:{:.2f}".format(slope))
        plt.savefig(cost_ratio_path[:-3]+"png")

        print("success")
        # plot the const cost part
        plt.figure()
        data = pd.read_csv(cost_const_path)
        x, y = list(data["HubConstCost"]), list(data["Values"])
        plt.plot(x, y, marker="o", ms=8)
        index = data.loc[data["HubConstCost"] == cfg.PARAM.HUB_BUILT_COST_CONST].index[0]
        x1, y1 = data.loc[index-1, ["HubConstCost", "Values"]]
        x2, y2 = data.loc[index+1, ["HubConstCost", "Values"]]

        # calculate the slope
        slope = (y2 - y1) / (x2 - x1)
        new_x = np.linspace(0.9*min(x), 1.1*max(x), 20)
        new_y = slope * (new_x - x1) + y1
        plt.plot(new_x, new_y, '--', linewidth=2)
        plt.xlabel("const cost")
        plt.ylabel("value")
        plt.title("Objective(ax + by)-HubConstCost Curve\n"+"Slope: {:.2f}".format(slope))
        plt.savefig(cost_const_path[:-3]+"png")

        # plot the vary cost part
        plt.figure()
        data = pd.read_csv(cost_vary_path)
        x, y = list(data["HubVaryCost"]), list(data["Values"])
        plt.plot(x, y, marker="o", ms=8)
        index = data.loc[data["HubVaryCost"] == cfg.PARAM.HUB_BUILT_COST_VARY].index[0]
        x1, y1 = data.loc[index-1, ["HubVaryCost", "Values"]]
        x2, y2 = data.loc[index+1, ["HubVaryCost", "Values"]]
        # calculate the slope
        slope = (y2 - y1) / (x2 - x1)
        new_x = np.linspace(0.9*min(x), 1.1*max(x), 20)
        new_y = slope * (new_x - x1) + y1
        plt.plot(new_x, new_y, '--', linewidth=2)
        plt.xlabel("vary cost")
        plt.ylabel("value")
        plt.title("Objective(ax + by)-HubVaryCost Curve\nSlope: {:.2f}".format(slope))
        plt.savefig(cost_vary_path[:-3]+"png")

        plt.show()

    def hub_cap_test(self):
        hub_cap_path = osp.join(self.base_dir, "cap.csv")

        index = np.where(cfg.EVALUATION.HUB_COST_VARY_FIELD == cfg.PARAM.HUB_CAPACITY)[0]
        cfg.EVALUATION.HUB_CAPACITY_FIELD = np.insert(cfg.EVALUATION.HUB_CAPACITY_FIELD, index, 0.95 * cfg.PARAM.HUB_CAPACITY)
        cfg.EVALUATION.HUB_CAPACITY_FIELD = np.insert(cfg.EVALUATION.HUB_CAPACITY_FIELD, index+2, 1.05 * cfg.PARAM.HUB_CAPACITY)

        new_dict = edict()
        num_hubs_list, amount_cost_list, time_cost_list = [], [], []
        mask = np.ones_like(cfg.EVALUATION.HUB_CAPACITY_FIELD)
        for idx, cap in enumerate(cfg.EVALUATION.HUB_CAPACITY_FIELD):
            new_dict["HUB_CAPACITY"] = cap
            merge_a_into_b(new_dict, cfg)
            solver = Solver()
            solver.solve(problem_id=3, tune_mode=True)
            with open(self.hub_path, "rb") as f:
                num_hubs = len(pkl.load(f))
            if num_hubs == 0:
                print("When capacity is :{}. No hubs were built...Continue...".format(cap))
                mask[idx] = 0
                continue
            self._pkl2csv()
            avg_amount_cost, avg_time_cost = self.get_avg_cost()
            amount_cost_list.append(avg_amount_cost)
            time_cost_list.append(avg_time_cost)
            num_hubs_list.append(num_hubs)
            print("Hub capacity: {:.0f} | Hubs: {:.0f} | Average amount cost: {:.3f}$ | Average time cost: {:.2f}m.".
                  format(cap, num_hubs, avg_amount_cost, avg_time_cost))
        df = pd.DataFrame(data=list(zip(cfg.EVALUATION.HUB_COST_VARY_FIELD[mask], num_hubs_list, amount_cost_list, time_cost_list)),
                          columns=["HubCapacity", "NumHubs", "AmountCost", "TimeCost"])
        df["Products"] = df["AmountCost"] * df["TimeCost"]
        df["Values"] = cfg.PARAM.WEIGHT_AMOUNT * df["AmountCost"] + cfg.PARAM.WEIGHT_TIME * df["TimeCost"]
        df.to_csv(hub_cap_path, index=False, float_format="%.2f")

    def hub_cap_plot(self):
        hub_cap_path = osp.join(self.base_dir, "cap.csv")
        plt.figure()
        data = pd.read_csv(hub_cap_path)
        x, y = list(data["HubCapacity"]), list(data["Values"])
        plt.plot(x, y, marker="o", ms=8)

        index = data.loc[data["HubCapacity"] == cfg.PARAM.HUB_BUILT_COST_CONST].index[0]
        x1, y1 = data.loc[index-1, ["HubCapacity", "Values"]]
        x2, y2 = data.loc[index+1, ["HubCapacity", "Values"]]
        slope = (y2 - y1) / (x2 - x1)
        new_x = np.linspace(0.9 * min(x1), 1.1 * max(x1), 20)
        new_y = slope * (new_x - x1) + y1
        plt.plot(new_x, new_y, '--')
        plt.xlabel("HubCapacity")
        plt.ylabel("Products")
        plt.title("Products(amountCost*timeCost)-HubCapacity Curve\nSlope: {:.2f}".format(slope))
        plt.savefig(hub_cap_path[:-3]+"png")
        plt.show()

    def _pkl2csv(self):
        """ convert some important data(useful for visualization) into csv format
        """
        with open(self.strategy_path, "rb") as f:
            data = pkl.load(f)
        time_costs = [sum(rec["TimeConsumption"]) for rec in data]
        amount_costs = [rec["AmountCost"] for rec in data]
        plane = [rec["Vehicles"].count("Plane") for rec in data]
        train = [rec["Vehicles"].count("Train") for rec in data]
        ship = [rec["Vehicles"].count("Ship") for rec in data]
        truck = [rec["Vehicles"].count("Truck") for rec in data]
        csv_dataset = list(zip(amount_costs, time_costs, plane, train, ship, truck))
        df = pd.DataFrame(csv_dataset, columns=["AmountCost", "TimeCost", "Plane", "Train", "Ship", "Truck"])
        df.to_csv(self.csv_path, index=False)
        print("Conversion from {} to {} finished.".format(self.strategy_path, self.csv_path))

    def plot_distribution(self):
        """
        view the distribution of results(e.g. vehicles, costs, time consumption)
        """
        data = pd.read_csv(self.csv_path)
        plt.figure()
        sns.distplot(data["AmountCost"])
        plt.xlabel("AmountCost")
        plt.ylabel("Frequency")
        plt.title("AmountCost(unit: $) Distribution")
        plt.savefig(osp.join(self.base_dir, "totalCost.png"))
        plt.figure()
        sns.distplot(data["TimeCost"])
        plt.xlabel("TimeCost(min)")
        plt.ylabel("Frequency")
        plt.title("TimeCost(unit: min) Distribution")
        plt.savefig(osp.join(self.base_dir, "timeCost.png"))
        vehicles = [data[vehicle].sum() for vehicle in cfg.VEHICLES]
        plt.figure()
        plt.pie(vehicles, labels=cfg.VEHICLES, autopct="%1.2f%%")
        plt.title("Vehicle Distribution")
        plt.savefig(osp.join(self.base_dir, "vehicle.png"))
        plt.show()

    def get_avg_cost(self):
        data = pd.read_csv(self.csv_path)
        length = len(data)

        avg_amount_cost = data["AmountCost"].sum() / length
        avg_time_cost = data["TimeCost"].sum() / length

        return round(avg_amount_cost, 2), round(avg_time_cost, 2)

    def weight_tune(self):
        """ tuning the weight of time and weight of cost
        """
        new_dict = edict()
        weight_data_path = osp.join(self.base_dir, "weight.csv")
        weight_time_list, weight_amount_list = [], []
        amount_cost_list, time_cost_list = [], []
        for weight_amount in cfg.EVALUATION.WEIGHT_AMOUNT_FIELD:
            for weight_time in cfg.EVALUATION.WEIGHT_TIME_FIELD:
                new_dict["WEIGHT_AMOUNT"] = round(weight_amount, 2)
                new_dict["WEIGHT_TIME"] = round(weight_time, 2)
                new_dict["NUM_ORDERS"] = 16
                # merge the config
                merge_a_into_b(new_dict, cfg)
                # solve the problem and get the output
                solver = Solver()
                solver.solve(problem_id=1, tune_mode=True)
                avg_amount_cost, avg_time_cost = self.get_avg_cost()
                weight_amount_list.append(round(weight_amount, 2))
                weight_time_list.append(round(weight_time, 2))
                amount_cost_list.append(avg_amount_cost)
                time_cost_list.append(avg_time_cost)
                print("When weight of amount is {:.2f} and weight of time is:{:.2f}".format(weight_amount, weight_time))
                print("Average amount cost is: {:.1f}. Average time cost is:{:.1f}".format(avg_amount_cost, avg_time_cost))
                # don't forget transform into csv file
                self._pkl2csv()

        df = pd.DataFrame(data=list(zip(weight_amount_list, weight_time_list, amount_cost_list, time_cost_list)),
                columns=["WeightOfAmount", "WeightOfTime", "AmountCost", "TimeCost"])
        # multiply the two columns to get the products
        df["Products"] = df["AmountCost"] * df["TimeCost"]
        df.to_csv(weight_data_path, index=False)

    def weight_plot(self):
        """
        draw a 3D-figure where x axis is weight_amount, y axis is weight_time, ,z axis is time_cost * amount * cost
        """
        weight_data_path = osp.join(self.base_dir, "weight.csv")
        weight_plot_path = osp.join(self.base_dir, "weight.png")
        # read data from txt we generated beforehand
        data = pd.read_csv(weight_data_path)
        # multiply the two columns to get the products
        x = data["WeightOfAmount"].unique().tolist()
        y = data["WeightOfTime"].unique().tolist()
        x, y = np.meshgrid(x, y)
        z = np.array(data["Product"]).reshape(x.shape)
        min_product = data["Product"].min()
        mask = data.loc[:, "Product"] == min_product
        best_a, best_b, amount, time = data[mask].iloc[0, :-1]

        # draw
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap="rainbow")
        ax.set_xlabel("Weight Of Amount Cost")
        ax.set_ylabel("Weight Of Time Cost")
        ax.set_zlabel("TimeCost * AmountCost")
        ax.set_title("Tuning parameters weight_time and weight_amount\nBest choice:({}, {}) => value: {}*{}={}".
            format(best_a, best_b, amount, time, min_product))
        plt.savefig(weight_plot_path)
        # plt.show()


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(prob_id=2, tune=True)
