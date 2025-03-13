import matplotlib.pyplot as plt
import numpy as np

# # exclude trivial rule def old weights
# ls = [0.4834247787,
# 0.4783772332,
# 0.4752454804,
# 0.4749122936,
# 0.4774588206,
# 0.4731637069,
# 0.46746466,
# 0.47727733]

# nls = [0.420372965,
# 0.358629999,
# 0.3542175431,
# 0.3494062462,
# 0.3574573842,
# 0.3535886266,
# 0.3513064968,
# 0.3598815731]

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(np.arange(1, 9, 1), ls, s=10, c='blue', marker="s", label='LS')
# ax1.scatter(np.arange(1, 9, 1), nls, s=10, c='skyblue', marker="s", label='NLS')
# plt.xlabel("b")
# plt.ylabel("error probability")
# plt.legend(fontsize=18)
# plt.savefig("table4_exclude_trivial_b1_8.png")
# plt.show()

# ls = [0.4602812653,
# 0.4377396531,
# 0.4063876456,
# 0.3760523226,
# 0.3472600801,
# 0.3222910798,
# 0.3013473739,
# 0.2858412714]

# nls = [0.3954180122,
# 0.3065487037,
# 0.2142407181,
# 0.136256193,
# 0.08715348771,
# 0.05795790758,
# 0.0399662142,
# 0.02890086436]

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(np.arange(1, 9, 1), ls, s=10, c='blue', marker="s", label='LS')
# ax1.scatter(np.arange(1, 9, 1), nls, s=10, c='skyblue', marker="s", label='NLS')
# plt.xlabel("epochs")
# plt.ylabel("error probability")
# plt.legend(fontsize=18)
# plt.savefig("table4_exclude_trivial_epochs1_8.png")
# plt.show()

# new weights 
# ls = [0.4503527457267048, 0.4335108271303277, 0.3878944688569754, 0.3489632887548456, 
#       0.3187687692973607, 0.3118772399082082, 0.2941931674132744, 0.3438079336424078]
# nls = [0.3698940242826938, 0.30291417648705343, 0.17011490927892742, 0.08594426191984289,  
#        0.057663413190360495,  0.062464301217939174, 0.03491571934559033, 0.06767419070375277]
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(np.arange(1, 9, 1), ls, s=10, c='blue', marker="s", label='LS')
# ax1.scatter(np.arange(1, 9, 1), nls, s=10, c='skyblue', marker="s", label='NLS')
# plt.xlabel("b")
# plt.ylabel("error probability")
# plt.legend(fontsize=18)
# plt.savefig("table4_new_weights.png")
# plt.show()

# Data epochs
ls = np.array([
    [0.455247841, 0.4548841982, 0.4550376033, 0.4679117397, 0.4683249443],
    [0.4288825171, 0.4290140204, 0.4307156398, 0.4506289625, 0.4494571254],
    [0.3952571775, 0.3971202246, 0.3971044958, 0.4203228201, 0.42213351],
    [0.3632105154, 0.3659407035, 0.3675446108, 0.3907375126, 0.3928282707],
    [0.3375221136, 0.3384048472, 0.3409292351, 0.3590657643, 0.3603784404],
    [0.3126494443, 0.3123002679, 0.3161948081, 0.334680276, 0.3356306027],
    [0.2905613387, 0.2939669546, 0.2989772569, 0.3110471452, 0.312184174],
    [0.2763270068, 0.2809881357, 0.2829203049, 0.2926683864, 0.2963025233]
])

nls = np.array([
    [0.3893737873, 0.3849487052, 0.3901725198, 0.4039925839, 0.4086024646],
    [0.2987967562, 0.2956264403, 0.2958791608, 0.3199977292, 0.3224434322],
    [0.2057392797, 0.2003403751, 0.199073707, 0.230720012, 0.2353302164],
    [0.127759596, 0.1264778539, 0.1232727771, 0.1507345977, 0.1530361402],
    [0.08126309031, 0.07991255293, 0.07929760934, 0.09722852743, 0.09806565853],
    [0.05418775674, 0.05251611654, 0.05233759074, 0.06329737132, 0.06745070254],
    [0.03848533465, 0.03655249407, 0.03616784996, 0.04361239688, 0.04501299543],
    [0.02661693335, 0.02634127307, 0.02622643358, 0.03311239357, 0.03220728826]
])

plt.figure(figsize=(10, 8))
data = ls
# Calculate mean and 95% confidence interval
mean_values = np.mean(data, axis=1)
ci_low = np.percentile(data, 2.5, axis=1)
ci_high = np.percentile(data, 97.5, axis=1)
# Plotting
x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o-', capsize=5, label='Concept LS')

data = nls
# Calculate mean and 95% confidence interval
mean_values = np.mean(data, axis=1)
ci_low = np.percentile(data, 2.5, axis=1)
ci_high = np.percentile(data, 97.5, axis=1)
# Plotting
x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o--', capsize=5, label='Concept NLS')

plt.xlabel('Epochs', fontsize=18)
plt.yticks([]) 
plt.xticks(fontsize=14)
# plt.ylabel('Error probability', fontsize=18)
plt.title('Prior-trained (varying epochs)', fontsize=24)
#plt.grid(True)
#plt.legend(fontsize=18)
plt.savefig('plots/table4_epochs.png')
plt.show()

# Data good model 
# ls = np.array([
#     [0.4543625114, 0.4543625114, 0.4543625114, 0.4543625114, 0.4543625114],
#     [0.424232226, 0.4186620491, 0.4199525664, 0.4253581989, 0.4197652498],
#     [0.390899242, 0.3858817456, 0.3785559539, 0.3836480106, 0.401186243],
#     [0.3839667996, 0.3898270122, 0.4068910992, 0.3862791136, 0.3839990259],
#     [0.354708461, 0.3622070068, 0.3503520512, 0.3680089753, 0.3444999045],
#     [0.3352812111, 0.3307707113, 0.3388625965, 0.3723223456, 0.3358065219],
#     [0.3384046073, 0.3323374956, 0.3367515796, 0.3443182455, 0.3305664138],
#     [0.3403405984, 0.3157249524, 0.329904579, 0.3908344445, 0.3273062863]
# ])

# nls = np.array([
#     [0.3783383428, 0.3718195386, 0.3814055178, 0.3875254727, 0.3703673271],
#     [0.3013102078, 0.2898517684, 0.2910206954, 0.2774956337, 0.2938057429],
#     [0.1844131541, 0.1670398504, 0.1566201471, 0.198043945, 0.2012186584],
#     [0.1319959829, 0.1406869713, 0.1041754475, 0.1811620509, 0.1432691565],
#     [0.1258101009, 0.1292041807, 0.08082136201, 0.1105093738, 0.08274022743],
#     [0.07765339177, 0.07808684645, 0.05497754745, 0.1118088857, 0.06328691561],
#     [0.09100105422, 0.1003496682, 0.09118530417, 0.08487736256, 0.06836721818],
#     [0.07739244116, 0.06899074778, 0.06945807524, 0.09654188525, 0.07094813884]
# ])

# plt.figure(figsize=(10, 8))
# data = ls
# # Calculate mean and 95% confidence interval
# mean_values = np.mean(data, axis=1)
# ci_low = np.percentile(data, 2.5, axis=1)
# ci_high = np.percentile(data, 97.5, axis=1)
# # Plotting
# x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
# plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o-', capsize=5, label='Concept LS')

# data = nls
# # Calculate mean and 95% confidence interval
# mean_values = np.mean(data, axis=1)
# ci_low = np.percentile(data, 2.5, axis=1)
# ci_high = np.percentile(data, 97.5, axis=1)
# # Plotting
# x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
# plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o--', capsize=5, label='Concept NLS')

# plt.xlabel('b', fontsize=18)
# plt.yticks([]) 
# plt.ylabel('Error probability', fontsize=18)
# plt.yticks(np.arange(0, 0.51, 0.1), fontsize=14)
# plt.xticks(fontsize=14) 
# plt.title('Prior-trained (varying b)', fontsize=24)
# # plt.grid(True)
# # plt.legend(fontsize=18)
# plt.savefig('plots/table4_b.png')
# plt.show()

# Data RR
# plt.figure(figsize=(10, 8))
# ls = np.array([[3.87, 3.86], [3.45,3.43], [3.3, 3.28], [3.18, 3.16], [2.96, 2.89], [2.62, 2.56], [2.26, 2.1], [2, 1.17]])
# nls = np.array([[3.62, 3.61], [2.93, 2.92], [2.42, 2.41], [1.97, 1.95], [1.56, 1.51], [1.13, 0.97], [0.48, 0.4], [0.12, 0.09]])

# ls /= 10
# nls /= 10

# data = ls
# # Calculate mean and 95% confidence interval
# mean_values = np.mean(data, axis=1)
# ci_low = np.percentile(data, 2.5, axis=1)
# ci_high = np.percentile(data, 97.5, axis=1)
# # Plotting
# x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
# plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o-', capsize=5, label='Concept LS')

# data = nls
# # Calculate mean and 95% confidence interval
# mean_values = np.mean(data, axis=1)
# ci_low = np.percentile(data, 2.5, axis=1)
# ci_high = np.percentile(data, 97.5, axis=1)
# # Plotting
# x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
# plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o--', capsize=5, label='Concept NLS')

# plt.xlabel('b', fontsize=18)
# # plt.ylabel('Error probability', fontsize=18)
# #plt.yticks(np.arange(0, 0.51, 0.1), fontsize=14)
# plt.yticks([])
# plt.ylim(0, 0.5)
# plt.xticks(fontsize=14)
# plt.title('Rational Rules', fontsize=24)
# # plt.grid(True)
# #plt.legend(fontsize=18)
# plt.savefig('plots/table4_rr.png')
# plt.show()

# Data humans

# ls = np.array([4.40, 4.25, 3.42, 2.88, 2.60, 1.74])
# nls = np.array([4.78, 3.44, 2.90, 2.18, 1.61, 0.88])

# ls /= 10
# nls /= 10

# data = ls
# # # Calculate mean and 95% confidence interval
# # mean_values = np.mean(data, axis=1)
# # ci_low = np.percentile(data, 2.5, axis=1)
# # ci_high = np.percentile(data, 97.5, axis=1)
# # # Plotting
# x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
# # plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o-', capsize=5, label='Concept LS')

# # data = nls
# # # Calculate mean and 95% confidence interval
# # mean_values = np.mean(data, axis=1)
# # ci_low = np.percentile(data, 2.5, axis=1)
# # ci_high = np.percentile(data, 97.5, axis=1)
# # # Plotting
# # x_values = np.arange(1, data.shape[0] + 1)  # Assuming each row represents a point
# # plt.errorbar(x_values, mean_values, yerr=[mean_values - ci_low, ci_high - mean_values], fmt='o--', capsize=5, label='Concept NLS')

# plt.figure(figsize=(10, 8))
# plt.plot(x_values, ls, 'o-', label='Concept LS')
# plt.plot(x_values, nls, 'o--', label='Concept NLS')
# plt.xlabel('Block', fontsize=18)
# # plt.ylabel('Error probability', fontsize=18)
# # plt.yticks(np.arange(0, 0.55, 0.1), fontsize=14)
# plt.xticks(fontsize=14)
# plt.ylabel('Error probability', fontsize=18)
# plt.yticks(np.arange(0, 0.51, 0.1), fontsize=14)
# plt.ylim(0, 0.5)
# plt.title('Humans', fontsize=24)
# # plt.grid(True)
# plt.legend(fontsize=18)
# plt.savefig('plots/table4_humans.png')
# plt.show()