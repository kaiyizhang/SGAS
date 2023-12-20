import matplotlib.pyplot as plt
import numpy as np


chair_path = '../log/Chair/part_comp//results/tmd_mmd.txt'
lamp_path = '../log/Lamp/part_comp//results/tmd_mmd.txt'
table_path = '../log/Table/part_comp//results/tmd_mmd.txt'

with open(chair_path, 'r') as f:
    lines = f.readlines()
    y = [float(i) for i in lines[1][:-1].split(' ')]
    item_1 = np.array([y for i in range(10)])
with open(lamp_path, 'r') as f:
    lines = f.readlines()
    y = [float(i) for i in lines[1][:-1].split(' ')]
    item_2 = np.array([y for i in range(10)])
with open(table_path, 'r') as f:
    lines = f.readlines()
    y = [float(i) for i in lines[1][:-1].split(' ')]
    item_3 = np.array([y for i in range(10)])

item = [item_1, item_2, item_3]
cat = ['chair', 'lamp', 'table']

for i in range(3):
    item = item[i]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.linspace(0.5, 5, 10)
    Y = np.linspace(1, 10, 10)
    X, Y = np.meshgrid(X, Y)
    Z1 = np.array(item)

    # Plot the surface.
    surf1 = ax.plot_surface(X, Y, Z1, cmap='Reds',
                        linewidth=0, antialiased=False)

    ax.tick_params(labelsize=14)
    ax.set_xlabel(r'$\tau_{mmd}$', size=22, labelpad=8)
    ax.set_ylabel(r'$\tau_{uhd}$', size=22, labelpad=8)
    ax.set_zlabel('TMD', size=20, labelpad=8)
    plt.savefig('tmd_mmd_uhd_%s.png' % (cat[i]), dpi=200)
