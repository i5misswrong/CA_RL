import matplotlib.pyplot as plt
import numpy as np

# x=np.linspace(0,300,5)
# y=np.linspace(0,1,5)
#
# x_tick=np.linspace(0,300,7)
# y_tick=np.linspace(0,1,11)
#
# plt.figure(figsize=(8,6))
# plt.xlim(0,300)
# plt.ylim(0,1)
# plt.xticks(x_tick,fontsize=14)
# plt.yticks(y_tick,fontsize=14)
# plt.plot(x,y)
# plt.show()


# x=np.linspace(0,100,5)
# y=np.linspace(0,90,5)
#
# x_tick=np.linspace(0,100,5)
# y_tick=np.linspace(0,90,10)
#
# plt.figure(figsize=(8,6))
# plt.xlim(0,100)
# plt.ylim(0,90)
# plt.xticks(x_tick,fontsize=14)
# plt.yticks(y_tick,fontsize=14)
# plt.plot(x,y)
# plt.show()


# x=np.linspace(550,1600,4)
# y=np.linspace(0,1,5)

x=[600, 800, 1000, 1400]
y=[89, 75, 67, 81]

x_tick=np.linspace(500,1500,11)
y_tick=np.linspace(20,100,9)

plt.figure(figsize=(8,6))
plt.xlim(600, 1500)
plt.ylim(20,100)
plt.xticks(x_tick,fontsize=14)
plt.yticks(y_tick,fontsize=14)
plt.plot(x,y, 'ks-')
plt.show()