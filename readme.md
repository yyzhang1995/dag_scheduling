# DAG任务调度（包括数据处理代码）

## graph_model

包括graph.py, building_graph.py 以及 utils.py三个文件

graph.py定义了Graph类

building_graph.py定义了从所给csv（需要将.xlsx转为.csv文件）生成DAG图和确定任务消耗时间的方法，此外还进行了一定的数据描述；

utils.py定义了一些工具函数，包括一些针对Graph类的函数

需要在项目目录下新建dataset文件夹并将.csv文件放入其中

## AOC

蚁群算法的初步实现（未完成）



另外由于我没有在新环境里配置项目，暂不提供requirement文档。所需要的三个重要的模块是numpy（1.19.5）, pytorch（1.7.1）以及pandas（1.1.5）