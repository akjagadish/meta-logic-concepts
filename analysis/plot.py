import matplotlib.pyplot as plt
import numpy as np

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

fixed_unmodified_distribution = [0.6712564229,
                             0.7280383879,
                             0.7983040326,
                             0.6435762484,
                             0.6373907757,
                             0.4275528127,
                             0.4535220939,
                             0.2851518786,
                             0.2093369106,
                             0.5935974799,
                             0.3869157903,
                             0.7910665293,
                             0.4155369055,
                             0.4289428924,
                             0.5829196228,
                             0.2552315087]

fixed_exclude_trivial_label = [0.6927333291,
                           0.7617306539,
                           0.8515776498,
                           0.6882728043,
                           0.6726076719,
                           0.4296592628,
                           0.4486984722,
                           0.2357242807,
                           0.1378123296,
                           0.6143555107,
                           0.373092456,
                           0.8456870958,
                           0.3992201525,
                           0.4013459287,
                           0.6028186033,
                           0.1872231711]

# variable features
variable_exclude_trivial_def = [0.6742536411,
                                  0.712016607,
                                  0.7606146379,
                                  0.6472681926,
                                  0.6183305804,
                                  0.4983634537,
                                  0.4711483307,
                                  0.3184052664,
                                  0.2351539566,
                                  0.5960987495,
                                  0.4596844365,
                                  0.7483777308,
                                  0.4353011984,
                                  0.4197715174,
                                  0.5823186865,
                                  0.2763351364]

variable_unmodified_distribution = [0.7033770011,
                                      0.7965241108,
                                      0.8663017794,
                                      0.6780541454,
                                      0.6578659298,
                                      0.4412713552,
                                      0.4340289824,
                                      0.2252058753,
                                      0.128841377,
                                      0.6238696562,
                                      0.413684134,
                                      0.8311442913,
                                      0.4383431731,
                                      0.3858458814,
                                      0.6189643928,
                                      0.1785294937]

variable_exclude_trivial_label = [0.6856360555,
                                    0.753229329,
                                    0.8601733243,
                                    0.6951333737,
                                    0.6766582224,
                                    0.4145923191,
                                    0.4189798299,
                                    0.2446286189,
                                    0.1435943717,
                                    0.6346055384,
                                    0.3893314803,
                                    0.8511924891,
                                    0.3921441661,
                                    0.405911586,
                                    0.6023131368,
                                    0.1853651497] 

# The data goes here - listing the values in order
rr_dnf = [0.82,0.81,0.92,0.61,0.61,0.47,0.47,0.21,0.07,0.57,0.44,0.95,0.44,0.28,0.57,0.13]
meta = fixed_exclude_trivial_def
random = [0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52]
human = [0.77,0.78,0.83,0.64,0.61,0.39,0.41,0.21,0.15,0.56,0.41,0.82,0.4,0.32,0.53,0.2]

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(human, meta, s=10, c='blue', marker="s", label='human vs MAML')
ax1.scatter(human, random, s=10, c='skyblue', marker="s", label='human vs Standard')
ax1.scatter(rr_dnf, meta, s=10, c='red', marker="o", label='RR vs MAML')
ax1.scatter(rr_dnf, random, s=10, c='lightcoral', marker="o", label='RR vs Standard')
plt.legend()
plt.savefig("fixed_exclude_trivial_def.png")
plt.show()

table4_fixed_exclude_trivial_def_LS = [0.4834247787,
                                        0.4783772332,
                                        0.4752454804,
                                        0.4749122936,
                                        0.4774588206,
                                        0.4731637069,
                                        0.46746466,
                                        0.47727733]

table4_fixed_exclude_trivial_def_NLS = [0.420372965,
                                        0.358629999,
                                        0.3542175431,
                                        0.3494062462,
                                        0.3574573842,
                                        0.3535886266,
                                        0.3513064968,
                                        0.3598815731]
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(np.arange(1, 9, 1), table4_fixed_exclude_trivial_def_LS, s=10, c='blue', marker="s", label='LS')
ax1.scatter(np.arange(1, 9, 1), table4_fixed_exclude_trivial_def_NLS, s=10, c='skyblue', marker="s", label='NLS')
plt.legend()
plt.savefig("table4_fixed_exclude_trivial_def.png")
plt.show()

table4_fixed_unmodified_LS = [0.4861608458,
                                0.4797613865,
                                0.4743832149,
                                0.4726364857,
                                0.4738876528,
                                0.4727167659,
                                0.4777601543,
                                0.4738454123]

table4_fixed_unmodified_NLS = [0.4124772549,
                                0.3753818871,
                                0.3691894278,
                                0.3636212557,
                                0.3716611015,
                                0.3666241344,
                                0.3688237121,
                                0.2971495982]

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(np.arange(1, 9, 1), table4_fixed_unmodified_LS, s=10, c='blue', marker="s", label='LS')
ax1.scatter(np.arange(1, 9, 1), table4_fixed_unmodified_NLS, s=10, c='skyblue', marker="s", label='NLS')
plt.legend()
plt.savefig("table4_fixed_unmodified.png")
plt.show()

table4_fixed_exclude_trivial_label_LS = [0.476439302,
                                        0.4747207958,
                                        0.476914976,
                                        0.4704520677,
                                        0.4737888863,
                                        0.4745946884,
                                        0.4727640964,
                                        0.4685865677]

table4_fixed_exclude_trivial_label_NLS = [0.4290255718,
                                            0.3672346128,
                                            0.3624882225,
                                            0.3354068618,
                                            0.3457568456,
                                            0.3550587405,
                                            0.3678071898,
                                            0.3533024155]

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.scatter(np.arange(1, 9, 1), table4_fixed_exclude_trivial_label_LS, s=10, c='blue', marker="s", label='LS')
ax1.scatter(np.arange(1, 9, 1), table4_fixed_exclude_trivial_label_NLS, s=10, c='skyblue', marker="s", label='NLS')
plt.legend()
plt.savefig("table4_fixed_exclude_trivial_label.png")
plt.show()