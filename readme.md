# DAG任务调度代码说明

## graph_model

包括graph.py, building_graph.py 以及 utils.py三个文件

graph.py定义了Graph类

building_graph.py定义了从所给csv（需要将.xlsx转为.csv文件）生成DAG图和确定任务消耗时间的方法，此外还进行了一定的数据描述；

utils.py定义了一些工具函数，包括一些针对Graph类的函数

需要在项目目录下新建dataset文件夹并将.csv文件放入其中

## AOC

蚁群算法的基础实现（未考虑起批时间）：aoc_dag.py

蚁群算法考虑起批时间（纯numpy实现，速度更快）：aoc_dag_with_bs_np.py

并行蚁群算法（目前结果有些问题，可能需要调试）：aoc_dag_with_bs_multi_process.py

## tools

主要是一些工具函数

可视化代码（各进程的任务以不同颜色的线段进行展示）：vision.py

计算量化指标：get_scores.py

将输出转换为.csv格式：change_form.py

# 运行环境

python3.7

## 安装

git clone https://github.com/yyzhang1995/dag_scheduling

或者下载.zip并解压

## 配置环境

pip install -r requirements.txt