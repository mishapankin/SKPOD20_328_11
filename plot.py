#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

fin = open("result_omp", "r")

x = []
y = []
z = []


for s in fin:
    l = s.split()
    if len(l) > 2:
        x.append(int(l[0]))
        y.append(int(l[1]))
        z.append(float(l[2]))

x = np.array(x)
y = np.array(y)
z = np.array(z)

ax = plt.axes(projection='3d')
ax.set_yticks([1, 8, 16, 32, 64])
ax.set_xlabel("Matrix size")
ax.set_ylabel("Thread count")
ax.set_zlabel("Time (s)")

ax.scatter3D(x, y, z)

plt.show()

# fin = open("result_omp_x86_2d", "r")

# x = []
# y = []


# for s in fin:
#     l = s.split()
#     x.append(int(l[1]))
#     y.append(float(l[2]))


# x = np.array(x)
# y = np.array(y)

# plt.xlabel("Thread count")
# plt.ylabel("Time (s)")
# plt.plot(x, y, 'ro')

# plt.show()

# fin = open("result_omp", "r")

# d = {}

# for s in fin:
#     l = s.split()
#     if len(l) > 0:
#         x = int(l[0])
#         try:
#             d[x]
#         except KeyError:
#             d[x] = []
#         # y = int(l[1])
#         z = 100000
#         if len(l) == 3:
#             z = float(l[2])
#         d[x].append(z)

# for k in d:
#     print(k, end=" & ")
#     for el in d[k]:
#         if el > 10000:
#             print("TL & ", end="")
#         else:
#             print("{:.02f}".format(el), end=" & ")
#     print("\\\\")