import pandas as pd
import numpy as np

y= []

for i in range(220500):
    val = "Column"+str(i+1)
    y.append(val)

y1 = np.array(y)
y1 = y1.reshape((1,220500))
df = pd.DataFrame(y1)
df.to_csv('collable.csv', index=False, header=False)
