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
**WARNING: This process may involve multiprocessing. Pay attention 
to your computational resources, as the process may be pretty slow and
computational intensive**
