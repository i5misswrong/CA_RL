from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
# r = np.linspace(0, 1.25, 50)
# p = np.linspace(0, 2*np.pi, 50)
# R, P = np.meshgrid(r, p)
# Z = ((R**2 - 1)**2)

x=np.linspace(-10,10,100)
y=np.linspace(-10,10,100)
print(x)
print(y)
X,Y=np.meshgrid(x,y)
z=np.exp(-((X)**2+(Y)**2)/30)
# Express the mesh in the cartesian system.
# X, Y = R*np.cos(P), R*np.sin(P)


# Plot the surface.
ax.plot_surface(X, Y, z, cmap=plt.cm.YlGnBu_r)

# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1.5)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
# ax.xticks=([])
# ax.yticks=([])
# ax.zticks=([])
ax.axis("off")
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')
ax.grid(False)
for i in range(100):

	
	if i>50:
		ax.view_init(elev=80-i*0.5, azim=i*1)
	else:
		ax.view_init(elev=20+i*0.5, azim=i*1)
	plt.savefig("%d.png"%(i+1))
	# plt.savefig("1.png")
# plt.show()