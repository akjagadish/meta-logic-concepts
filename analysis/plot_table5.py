import matplotlib.pyplot as plt
import numpy as np

# vary b old weights
# b1 = [21.90819829 ,37.83427223, 32.90587645, 31.24234203, 38.33630775, 47.54308037]
# b2 = [6.76288105,32.75958001,23.67546237,22.85787184,29.93678777,46.82489623]
# b3 = [3.100341292,27.80310197,19.98791128,19.18993414,26.33394898,45.88086243]
# b4 = [1.25386288,29.06149419,21.7311264,16.81909154,27.58784545,45.736377]
# b5 = [1.086402885,30.96421661,24.00778029,17.59395212,32.35517163,45.49684947]
# b6 = [1.078450339,35.6040541,25.07208143,19.68658686,35.71846874,47.12765736]
# b7 = [0.9771603882,36.67443692,28.12165343,19.63381535,38.21699827,47.22525929]
# b8 = [0.7978095898,37.21370282,26.87843556,20.5748961,37.17988941,47.43172824]

# b new weights with skip connections
b1 = [0.2408288335148244,
0.4271927445568144,
0.33788776262663306,
0.332781201559119,
0.4001648526079953,
0.4961234012618662]

b2 = [0.08029696719255294,
0.29939749790530185,
0.16592417959705924,
0.21019632605079094,
0.2887885234650457,
0.48615936198271803]

b3 = [0.013521755931139983,
0.09458382843565687,
0.1014877910738551,
0.13474826708910884,
0.22542355609145387,
0.455876424729795]

b4 = [0.00734346137914539,
0.0434528156852993,
0.031785420361518844,
0.059254148166009095,
0.12111455065834964,
0.31946839486307005]

b5 = [0.009507668601721031,
0.02258856910545153,
0.028636374457865158,
0.039928567468495706,
0.05390243102360919,
0.18020253092894561]

b6 = [0.0038405937875432518,
0.008016160356676362,
0.005132642991663434,
0.008017859601886067,
0.015000166041311457,
0.07566321794545715]

b7 = [0.004318242347436304,
0.008425991752358884,
0.0056970669675334654,
0.006624710916227132,
0.011365977336252976,
0.04225668991657598]

b8 = [0.006009524235878571,
0.010027993775008492,
0.009316459960779585,
0.01613318774227836,
0.016004722732578136,
0.08040226985406686]

b1 = [x for x in b1]
b2 = [x for x in b2]
b3 = [x for x in b3]
b4 = [x for x in b4]
b5 = [x for x in b5]
b6 = [x for x in b6]
b7 = [x for x in b7]
b8 = [x for x in b8]

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot()
ax1.plot(np.arange(1, 7, 1), b1, linestyle='-', color='blue', marker="s", label='b=1')
ax1.plot(np.arange(1, 7, 1), b2, linestyle='-', color='red', marker="s", label='b=2')
ax1.plot(np.arange(1, 7, 1), b3, linestyle='-', color='gold', marker="s", label='b=3')
ax1.plot(np.arange(1, 7, 1), b4, linestyle='-', color='green', marker="s", label='b=4')
ax1.plot(np.arange(1, 7, 1), b5, linestyle='-', color='purple', marker="s", label='b=5')
ax1.plot(np.arange(1, 7, 1), b6, linestyle='-', color='orange', marker="s", label='b=6')
ax1.plot(np.arange(1, 7, 1), b7, linestyle='-', color='turquoise', marker="s", label='b=7')
ax1.plot(np.arange(1, 7, 1), b8, linestyle='-', color='hotpink', marker="s", label='b=8')
plt.xlabel("Concept", fontsize=18)
plt.ylabel("Error probability", fontsize=18)
plt.xticks(ticks=np.arange(1, 7), labels=['I', 'II', 'III', 'IV', 'V', 'VI'], fontsize=20)
plt.ylim(0, 0.5)
plt.yticks(fontsize=20)
plt.legend(fontsize=12)
plt.title('Prior-trained (varying b)', fontsize=24)
plt.savefig("plots/table5_b.png")
plt.show()

# e1=[21.9259302,
# 38.01208432,
# 32.89874949,
# 31.22248238,
# 38.10494229,
# 47.83894874]
# e2=[11.44724641,
# 24.75993671,
# 22.01322918,
# 19.23475261,
# 29.59206637,
# 45.5547239]
# e3=[6.58158002,
# 15.20667185,
# 13.6936853,
# 11.11553641,
# 22.98497696,
# 43.35917335]
# e4=[4.013062017,
# 9.42718135,
# 8.447270601,
# 7.082282969,
# 17.28836786,
# 40.89988657]
# e5=[2.673123393,
# 6.235098652,
# 5.186077743,
# 4.124861042,
# 12.20051069,
# 37.64460446]
# e6=[1.896149858,
# 4.474895616,
# 3.552394152,
# 3.298567558,
# 8.114642107,
# 32.93805667]
# e7=[1.34914878,
# 3.253325519,
# 2.737280774,
# 2.698562237,
# 6.287541741,
# 27.74676569]
# e8=[1.102205095,
# 2.38175278,
# 1.987486664,
# 1.728824987,
# 4.349396013,
# 22.1718962]

# fig = plt.figure(figsize=(10, 8))
# ax1 = fig.add_subplot()
# ax1.plot(np.arange(1, 7, 1), e1, linestyle='-', color='blue', marker="s", label='epoch 1')
# ax1.plot(np.arange(1, 7, 1), e2, linestyle='-', color='red', marker="s", label='epoch 2')
# ax1.plot(np.arange(1, 7, 1), e3, linestyle='-', color='gold', marker="s", label='epoch 3')
# ax1.plot(np.arange(1, 7, 1), e4, linestyle='-', color='green', marker="s", label='epoch 4')
# ax1.plot(np.arange(1, 7, 1), e5, linestyle='-', color='purple', marker="s", label='epoch 5')
# ax1.plot(np.arange(1, 7, 1), e6, linestyle='-', color='orange', marker="s", label='epoch 6')
# ax1.plot(np.arange(1, 7, 1), e7, linestyle='-', color='turquoise', marker="s", label='epoch 7')
# ax1.plot(np.arange(1, 7, 1), e8, linestyle='-', color='hotpink', marker="s", label='epoch 8')
# plt.xlabel("Concept", fontsize=18)
# plt.xticks(ticks=np.arange(1, 7), labels=['I', 'II', 'III', 'IV', 'V', 'VI'], fontsize=20)
# plt.yticks([])
# plt.title('Prior-trained (varying epochs)', fontsize=24)
# # plt.ylabel("error probability")
# plt.legend(fontsize=12)
# plt.savefig("plots/table5_epochs.png")
# plt.show()