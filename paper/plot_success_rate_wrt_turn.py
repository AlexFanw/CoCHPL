import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (14, 3)

lastfm_scpr = [0.709, 0.773, 0.829, 0.844, 0.852]
lastfm_unicorn = [0.788, 0.854, 0.884, 0.901, 0.911]
lastfm_LSOPL = [0.802, 0.894, 0.96, 0.977, 0.986]
yelp_scpr = [0.489, 0.535, 0.554, 0.57, 0.574]
yelp_unicorn = [0.52, 0.568, 0.593, 0.612, 0.624]
yelp_LSOPL = [0.541, 0.623, 0.699, 0.755, 0.819]

x = [1, 2, 3, 4, 5]

plt.subplot(1, 2, 1)
plt.plot(x, lastfm_scpr, label="SCPR", marker='^', ms=8)
plt.plot(x, lastfm_unicorn, label="UNICORN", marker='x', ms=8)
plt.plot(x, lastfm_LSOPL, label="HLSOPL", marker='o', ms=8)
plt.xlabel("Number of Asked Attributes", fontsize=18, weight='bold')
plt.ylabel("Success Rate@15", fontsize=18, weight='bold')
plt.xticks(fontsize=16)
plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], fontsize=16)
plt.legend(frameon=False, loc=2, fontsize=11, prop={'weight': 'bold'})
plt.title("LastFM", fontsize=18, weight='bold')

plt.subplot(1, 2, 2)
plt.plot(x, yelp_scpr, label="SCPR", marker='^', ms=8)
plt.plot(x, yelp_unicorn, label="UNICORN", marker='x', ms=8)
plt.plot(x, yelp_LSOPL, label="HLSOPL", marker='o', ms=8)
plt.xlabel("Number of Asked Attributes", fontsize=18, weight='bold')
# plt.ylabel("Success Rate@15", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks([0.45, 0.52, 0.59, 0.66, 0.73, 0.80, 0.87], fontsize=16)
plt.legend(frameon=False, loc=2, fontsize=11, prop={'weight': 'bold'})
plt.title("Yelp", fontsize=18, weight='bold')

plt.savefig('./plot_success_rate_wrt_turn.pdf', dpi=800, bbox_inches='tight')
