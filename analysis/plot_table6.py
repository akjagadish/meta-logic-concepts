import matplotlib.pyplot as plt
import numpy as np

humans_b1 = [0.64,0.64,0.66,0.55,0.57,0.43,0.46,0.34,0.46,0.41,0.52,0.5,0.73,0.59,0.39,0.46]
humans_b7 = [0.96,
0.93,
1,
0.96,
0.02,
0,
0.05,
0,
0.66,
0.64,
0.64,
0.66,
0.36,
0.36,
0.27,
0.3]

rr_b1 = [0.84,0.54,0.84,0.54,0.46,0.16,0.46,0.16,0.2,0.2,0.5,0.5,0.8,0.8,0.5,0.5]
rr_b7 = [1,1,1,0.99,0,0,0.01,0,0.56,0.55,0.57,0.56,0.45,0.44,0.44,0.43]

nn_b1 = [0.7874543905,
0.6489578022,
0.7849718823,
0.6135294982,
0.386253921,
0.21973242,
0.3272625547,
0.2057096729,
0.2544609297,
0.2970084384,
0.4343620155,
0.5138066064,
0.731515169,
0.7206745213,
0.5250438674,
0.5260738005]

nn_b7 = [0.8599327712,
0.7077895899,
0.8445248706,
0.7021879143,
0.3127272444,
0.1580987289,
0.3101749285,
0.1524380608,
0.2719872707,
0.3368937837,
0.4470340307,
0.5156986222,
0.6883925554,
0.7241738291,
0.5047640358,
0.5488719442]

nn_7epochs = [0.9996848654,
0.9984214685,
0.9996778369,
0.9978605558,
0.003035927095,
0.002134786119,
0.002479761561,
0.00175890101,
0.02572221347,
0.05231556423,
0.108534435,
0.34237459,
0.9466089913,
0.961886956,
0.7762772864,
0.872506426]


fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(humans_b1, rr_b1, s=10, c='red', marker="o")
plt.xlabel("Human")
plt.ylabel("RR")
 
plt.savefig("table6 RR vs Human b=1")
plt.show()
print("RR vs Human b=1", np.corrcoef(humans_b1, rr_b1)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(humans_b7, rr_b7, s=10, c='red', marker="o")
plt.xlabel("Human")
plt.ylabel("RR")
 
plt.savefig("table6 RR vs Human b=7")
plt.show()
print("RR vs Human b=7", np.corrcoef(humans_b7, rr_b7)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(rr_b1, nn_b1, s=10, c='red', marker="o")
plt.xlabel("NN")
plt.ylabel("RR")
 
plt.savefig("table6 RR vs NN b=1")
plt.show()
print("RR vs NN b=1", np.corrcoef(rr_b1, nn_b1)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(rr_b7, nn_b7, s=10, c='red', marker="o")
plt.xlabel("NN")
plt.ylabel("RR")
 
plt.savefig("table6 RR vs NN b=7")
plt.show()
print("RR vs NN b=7", np.corrcoef(rr_b7, nn_b7)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(rr_b7, nn_7epochs, s=10, c='red', marker="o")
plt.xlabel("NN")
plt.ylabel("RR")
 
plt.savefig("table6 RR vs NN 7 epochs")
plt.show()
print("RR vs NN 7epochs", np.corrcoef(rr_b7, nn_7epochs)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(humans_b1, nn_b1, s=10, c='red', marker="o")
plt.xlabel("human")
plt.ylabel("nn")
 
plt.savefig("table6 NN vs Humans b=1")
plt.show()
print("NN vs Humans b=1", np.corrcoef(humans_b1, nn_b1)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(humans_b7, nn_b7, s=10, c='red', marker="o")
plt.xlabel("human")
plt.ylabel("nn")
 
plt.savefig("table6 NN vs Humans b=7")
plt.show()
print("NN vs Humans b=7", np.corrcoef(humans_b7, nn_b7)[0, 1])

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(humans_b7, nn_7epochs, s=10, c='red', marker="o")
plt.xlabel("human")
plt.ylabel("nn")
 
plt.savefig("table6 NN vs Humans 7 epochs")
plt.show()
print("NN vs Humans 7epochs", np.corrcoef(humans_b7, nn_7epochs)[0, 1])