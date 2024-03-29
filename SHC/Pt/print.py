import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('Omega.dat')
k_nodes=[]
label=[]
f=open('omega_klabel')
for i in f.readlines():
    k_nodes.append(float(i.split()[0]))
    label.append(i.split()[1])
fig,ax=plt.subplots()
ax.plot(data[:,0],data[:,1:])
for x in k_nodes:
    ax.axvline(x,c='k')
ax.set_xticks(k_nodes)
ax.set_xticklabels(label)
ax.set_xlim([0,k_nodes[-1]])
fig.savefig('omega.pdf')
