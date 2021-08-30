import pandas as pd

m = 120
file = f'../results/working_order_m={m}.txt'

with open(file, 'r') as f:
    res = f.readlines()

flows = []
for i in range(m):
    flows.append(res[2 * i + 1].split(','))

lengths = [len(flow) for flow in flows]
max_length = max(lengths)
for i in range(m):
    flows[i] += ([''] * (max_length - len(flows[i])))

columns = [f'machine_{i + 1}' for i in range(m)]
df = pd.DataFrame({column: flow for (column, flow) in zip(columns, flows)})
df.to_csv(f'../results/working_order_m={m}.csv')
