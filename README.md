# PACKAGE-DELIVERY
* As a speed up, we convert the official offered \*.xlsx data into \*.csv format data.
* We split the TableD.xlsx into 4 sub-csvs(train.csv...)
* As another speedup, we parse all the records about transportation and construct the graph. PKL model is available here [Model](http://resources.dbgns.com/package-delivery/graph.pkl)

# Usage 
```
git clone https://github.com/dbgns/package-delivery ~
cd ~/package-delivery
wget http://resources.dbgns.com/package-delivery/graph.pkl data/
cd src
python graph.py
```
The last command will create serveral processes to parse the orders. <br>
**WARNING:** Pay attention to your computational resources, as the process is pretty computational intensive and slow.
