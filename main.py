import torch as th
import tensorflow as tf
import datetime as dt
import matplotlib.pyplot as plt

#%%

x = th.rand(2,2)
print(x)
print(th.cuda.is_available())
print(th.rand(2,2).cuda())
d = dt.date(year = 2021, month = 1, day = 1)
print(d.month)
print(d.strftime('%Y %m%d'))
plt.plot([4,2,3,7,1])