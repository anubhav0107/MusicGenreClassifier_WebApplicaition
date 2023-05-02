import pandas as pd

y= []

for i in range(100):
    val = "Bsong"+str(i+1)
    y.append(val)

df = pd.DataFrame(y)
df.to_csv('blabel.csv', index=False, header=False)
