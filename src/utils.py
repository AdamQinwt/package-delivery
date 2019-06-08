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


def std2sec(std_time):
    """ transform std format time into seconds(e.g. 17:31:50 => 17 * 60 * 60 + 31 * 60 + 50)
    :param std_time: std format time
    :return:
    """
    hour, minute, second = std_time.split(":")
    return int(hour) * 3600 + int(minute) * 60 + int(second)


def sec2std(sec_time):
    """ transform second format time into std format time
    :param sec_time:
    :return:
    """
    day = 0
    hour = int(sec_time // 3600)
    while hour >= 24:
        hour -= 24
        sec_time -= 24 * 3600
        day += 1
    minute = int((sec_time - hour * 3600) // 60)
    second = int(sec_time - hour * 3600 - minute * 60)
    hour, minute, second = str(hour), str(minute), str(second)

    if len(minute) == 1:
        minute = '0' + minute
    if len(second) == 1:
        second = '0' + second
    # if we need more than one day
    if day != 0:
        return ':'.join(['(+' + str(day) + 'day)', hour, minute, second])
    return ':'.join([hour, minute, second])


def xlsx2csv(filename, with_header=True):
    """ convert the .xlsx format data into csv format(speed up I/O)
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
    """ do some pre-processing work
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
