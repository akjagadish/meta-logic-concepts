import matplotlib.pyplot as plt
import numpy as np

# for b = 2
fixed_exclude_trivial_def = [0.7107010416, 
                    0.7671524207,
                    0.8439980453,
                    0.6908307576,
                    0.7070753214,
                    0.4077716437,
                    0.4591202334,
                    0.221756097,
                    0.1415421804,
                    0.5686388402,
                    0.3430387968,
                    0.8448546216,
                    0.4175972141,
                    0.3928242921,
                    0.6043098338,
                    0.1965197032]

rr_dnf = [0.82,0.81,0.92,0.61,0.61,0.47,0.47,0.21,0.07,0.57,0.44,0.95,0.44,0.28,0.57,0.13]
meta = fixed_exclude_trivial_def
random = [0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52]
human = [0.77,0.78,0.83,0.64,0.61,0.39,0.41,0.21,0.15,0.56,0.41,0.82,0.4,0.32,0.53,0.2]

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(rr_dnf, meta, s=10, c='red', marker="o", label='MAML')
# ax1.scatter(rr_dnf, human, s=10, c='blue', marker="o", label='Human')
# plt.xlabel("RR predictions")
# plt.ylabel("Humans/NN with b=2")
# plt.legend()
# plt.savefig("table3_fixed_exclude_trivial_def.png")
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(human, meta, s=10, c='blue', marker="s", label='MAML')
# ax1.scatter(human, random, s=10, c='red', marker="s", label='Standard')
# plt.xlabel("Human predictions")
# plt.ylabel("NN")
# plt.legend()
# plt.savefig("plots/table3_humans.png")

# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(rr_dnf, meta, s=10, c='blue', marker="o", label='MAML')
# ax1.scatter(rr_dnf, random, s=10, c='red', marker="o", label='Standard')
# plt.xlabel("Rational Rules predictions")
# plt.legend()
# plt.savefig("plots/table3_RR.png")
# plt.show()

# Plotting
# Plotting
x_ticks = np.arange(0, 1.2, 0.2)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot 1
axs[0].scatter(human, meta, s=50, c='blue', marker="o", label='Prior-trained')  # Increase marker size
axs[0].scatter(human, random, s=50, c='red', marker="o", label='Standard')  # Increase marker size
axs[0].set_xlabel("Human predictions", fontsize=18)  # Increase font size
axs[0].set_ylabel("Neural Network", fontsize=18)  # Increase font size
axs[0].set_xticks(x_ticks)  # Set x-axis ticks
axs[0].tick_params(axis='both', which='major', labelsize=14)  # Increase font size for ticks
axs[0].legend(fontsize=18)  # Increase font size for legend

# Plot 2
axs[1].scatter(rr_dnf, meta, s=50, c='blue', marker="o", label='Prior-trained')  # Increase marker size
axs[1].scatter(rr_dnf, random, s=50, c='red', marker="o", label='Standard')  # Increase marker size
axs[1].set_xlabel("Rational Rules predictions", fontsize=18)  # Increase font size
axs[1].set_xticks(x_ticks)  # Set x-axis ticks
axs[1].tick_params(axis='both', which='major', labelsize=14)  # Increase font size for ticks
axs[1].set_yticks([])  # Hide y ticks for the second plot
axs[1].legend(fontsize=18)  # Increase font size for legend

plt.tight_layout()
plt.savefig("table3_combined.png")
plt.show()