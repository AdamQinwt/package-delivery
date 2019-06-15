#!/usr/bin/env python
# coding=utf-8

# Author: Junjie Wang
# Mail: dreamboy.gns@sjtu.edu.cn

# Website:http://www.dbgns.com
# Blog:http://www.dbgns.com/blog

"""
some help functions
"""

import pandas as pd
import os.path as osp

DATA_DIR = "../data"
TABLE = {"TableA": "order", "TableB": "commodity", "TableC": "distance", "TableD": "vehicle"}
VEHICLES = ['Plane', 'Ship', 'Truck', 'Train']


def std2min(std_time):
    """
    convert the time of standard form into minutes

    :param std_time:  time in the standard form(e.g. 22:41:30)
    :return: minutes
    """
    hour, minute, second = std_time.split(":")
    return int(hour) * 60 + int(minute) + round(int(second)/60, 2)


def min2std(min_time, get_day=False):
    """
    convert the time in the form of minutes into standard form

    :param min_time: minutes
    :param get_day: calculate day or not(e.g. (+1day)17:31:25, useful when we want to get true arrival time
    :return: the time in the standard form
    """
    day = 0
    second = int((min_time - int(min_time)) * 60)
    min_time = int(min_time)
    hour = min_time // 60
    while hour >= 24:
        day += 1
        hour -= 24
    minute = min_time % 60
    hour, minute, second = str(hour), str(minute), str(second)

    if len(minute) == 1:
        minute = '0' + minute
    if len(second) == 1:
        second = '0' + second
    # if we need more than one day
    if day != 0 and get_day:
        return ':'.join(['(+' + str(day) + 'day)', hour, minute, second])
    return ':'.join([hour, minute, second])


def xlsx2csv(filename, with_header=True):
    """
    convert the .xlsx format data into csv format(speed up I/O)

    :param filename: xlsx filename
    :param with_header(e.g. the distance.xlsx does not have a header)
    """
    csv_path = osp.join(DATA_DIR, TABLE[osp.splitext(filename)[0]] + ".csv")

    if osp.exists(csv_path):
        print("Found {0}. No need for conversion.".format(csv_path))
        return

    if with_header:
        xlsx = pd.read_excel(osp.join(DATA_DIR, filename))
    else:
        xlsx = pd.read_excel(osp.join(DATA_DIR, filename), header=None)

    xlsx.to_csv(csv_path, index=False)
    print("Saving => {0}.".format(csv_path))
    return


def prep_data():
    """
    do some pre-processing work

    """
    xlsx2csv("TableA.xlsx")
    xlsx2csv("TableB.xlsx")
    xlsx2csv("TableC.xlsx", with_header=False)

    order = pd.read_csv(osp.join(DATA_DIR, "order.csv"))
    order.columns = ['CityOfSeller', 'CityOfPurchaser', 'TimeOfOrder', 'IndexOfCommodity', 'AmountOfCommodity', 'Emergency']
    order.to_csv(osp.join(DATA_DIR, "order.csv"), index=False)

    commodity = pd.read_csv(osp.join(DATA_DIR, "commodity.csv"))
    commodity.columns = ['IndexOfCommodity', 'PriceOfUnit', 'CategoryOfCommodity', 'Weight']
    commodity.to_csv(osp.join(DATA_DIR, "commodity.csv"), index=False)

    if osp.exists(osp.join(DATA_DIR, "vehicle.csv")):
        print("Found {0}. No need for conversion.".format(osp.join(DATA_DIR, "vehicle.csv")))
        return

    vehicles_dict = {}
    for sheet_name in VEHICLES:
        vehicles_dict[sheet_name] = pd.read_excel(osp.join(DATA_DIR, "TableD.xlsx"), sheet_name=sheet_name)
        vehicles_dict[sheet_name]['Vehicle'] = sheet_name

    vehicles = pd.concat(vehicles_dict.values())
    vehicles.columns = ['IndexOfDepartureCity', 'IndexOfArrivalCity',
                        'AverageDelay', 'Speed', 'Cost', 'TimeOfDeparture', 'Vehicle']
    vehicles.to_csv(osp.join(DATA_DIR, "vehicle.csv"), index=False)
    print("Vehicles schedule merged into {0}".format(osp.join(DATA_DIR, "vehicle.csv")))


def look_up_vehicle(seller_city, purchaser_city):
    vehicle = pd.read_csv(osp.join(DATA_DIR, "vehicle.csv"))

    if vehicle.loc[(vehicle['IndexOfDepartureCity'] == seller_city)
                    & (vehicle['IndexOfArrivalCity'] == purchaser_city)].empty:
        return None
    else:
        return vehicle.loc[(vehicle['IndexOfDepartureCity'] == seller_city)
                    & (vehicle['IndexOfArrivalCity'] == purchaser_city)]


if __name__ == '__main__':
    prep_data()
