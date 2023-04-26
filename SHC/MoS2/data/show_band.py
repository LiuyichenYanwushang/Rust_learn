from new_pythtb import *
import numpy as np
import matplotlib.pyplot as plt


A=w90("./","wannier90")
zero_energy=-2.0
model=A.model(zero_energy)
path=[[0.0000,0.0000,0.0],[0.5000,0.0000,0.000],[0.66666,0.33333,0.0],[0.0,0.0,0.0]]
(kvec,kdist,knode)=model.k_path(path,101)
band=model.solve_all(kvec)
fig,ax=plt.subplots()
ax.plot(kdist,band,c='k')
fig.savefig("band.pdf")
