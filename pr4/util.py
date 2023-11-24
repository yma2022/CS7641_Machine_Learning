from matplotlib import pyplot as plt
import numpy as np

def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
	
	decay_steps = int(max_steps * decay_ratio)
	rem_steps = max_steps - decay_steps
	values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
	values = (values - values.min()) / (values.max() - values.min())
	values = (init_value - min_value) * values + min_value
	values = np.pad(values, (0, rem_steps), 'edge')
	return values


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
	font_size = 'x-large'
	if policy.shape[1] > 16:
		font_size = 'small'
	plt.title(title)
	for i in range(policy.shape[0]):
		for j in range(policy.shape[1]):
			y = policy.shape[0] - i - 1
			x = j
			p = plt.Rectangle([x, y], 1, 1)
			p.set_facecolor(color_map[map_desc[i,j]])
			ax.add_patch(p)
			if color_map[map_desc[i,j]] != 'black':
				text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
							horizontalalignment='center', verticalalignment='center', color='w')
	plt.axis('off')
	plt.xlim((0, policy.shape[1]))
	plt.ylim((0, policy.shape[0]))
	plt.tight_layout()
	plt.savefig('img/' + title+str('.png'))
	plt.close()


def plot_blackjack_policy(policy, title):
    plt.imshow(np.expand_dims(policy, axis=0), cmap='coolwarm', origin='lower', aspect='auto', extent=[1, 11, 12, 21])
    plt.colorbar(label='Action (0: Stick, 1: Hit)')
    plt.xlabel("Dealer's Face-up Card")
    plt.ylabel("Player's Sum")
    plt.title(title)
    plt.xticks(np.arange(1, 12, 1))
    plt.yticks(np.arange(12, 22, 1))
    plt.show()