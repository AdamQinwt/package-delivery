# Package-Delivery
> Project1 of Algroithms And Complexity(SJTU. 2019. Spring)

## Introduction
&nbsp;&nbsp;In this project, we need to help the Express Company schedule the orders on "Double 11" Day.
Officially offered data are available [here(*.xlsx)](data).There are 4 different scenarios in total
(refer to the [problem.pdf](report/problem.pdf))for more details
* All of the 656 cities covered in the orders have stations
Design an efficient algorithm to transfer packages
* Hubs can transfer packages with lower unit cost, but a specific
hub can only use one transportation tool to transfer packages to each neighbour.
Design the scheme to build the hubs
* There may be some other constraints(e.g. Inflammable commodities
will be rejected by the hubs and can not be transported by
airlines). Besides, the hubs are capacitied. Revise the model
you designed above.
* If only large cities(cities with airline service) have station, 
how to schedule the packages efficiently?

## Demo
``` shell
git clone https://github.com/dbgns/package-delivery ~
echo "export PYTHONPATH=~/package-delivery/:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

cd ~/package-delivery/src
python demo.py --problem_id 1 --num_orders 2400 --num_processes 16
```
## Default Parameters
|   Parameter    |     Value     |
|     :--:       |      :--:     |
| WeightOfAmount |      15       |
| WeightOfTime   |       1       |
|  HubCapacity   |      1000     |
| HubCostRatio   |       0.7     |
| HubConstCost   |      3000     |
|  HubVaryCOst   |       300     |
## Performance Evaluation
#### Average Cost
&nbsp;&nbsp;Using the paramters listed above, we run our model for 2400 orders and 240000 orders respectively.
|           |                     2400 orders                     |                 240000 orders                    |
| problem1  |   AverageAmountCost($)  |   AverageTimeCost(min)    |  AverageAmountCost($)  |  AverageTimeCost(min)   |
| :--:      |         :--:            |        :--:               |       :--:             |      :--:               |
![trade-off](imgs/tune/cost-rate.png)
![weight-tune](imgs/tune/weight.png)
![capacity]()
![const-cost]()
![vary-cost]()
![ratio]()
**WARNING: This process may involve multiprocessing. Pay attention 
to your computational resources, as the process may be pretty slow and
computational intensive**
