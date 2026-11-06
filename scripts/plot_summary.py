"""Create figures from the saved model-generated tables."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def rows(name):
    with (ROOT / 'data' / name).open() as f:
        return list(csv.DictReader(f))

fig, ax = plt.subplots(figsize=(8.5, 4.8), layout='constrained')
for name, label, color in [('Discrete', 'Discrete monitoring', '#c75b39'),
                           ('Bridge', 'Brownian bridge', '#246b9b')]:
    selected = [r for r in rows('barrier_mc.csv') if r['method'] == name and int(r['N']) == 200000]
    ax.errorbar([int(r['m']) for r in selected], [float(r['price']) for r in selected],
                yerr=[1.96 * float(r['se']) for r in selected], marker='o', capsize=4,
                label=label, color=color)
ax.axhline(2.223538991350488, color='#333333', ls='--', label='Independent reference')
ax.set_xscale('log', base=4)
ax.set_xticks([1, 4, 16, 64, 256], ['1', '4', '16', '64', '256'])
ax.set(xlabel='Monitoring intervals', ylabel='Up-and-out option price',
       title='More paths cannot fix missed barrier crossings')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.2)
fig.savefig(ROOT / 'figures/barrier_monitoring.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.8), layout='constrained')
for name, label, color in [('Standard', 'Standard Monte Carlo', '#c75b39'),
                           ('Geometric CV', 'Geometric control variate', '#246b9b')]:
    selected = [r for r in rows('asian_mc.csv') if r['method'] == name]
    ax.loglog([int(r['N']) for r in selected], [float(r['se']) for r in selected],
              'o-', label=label, color=color)
ax.set(xlabel='Production paths', ylabel='Monte Carlo standard error',
       title='Asian pricing: lower sampling error with a geometric control')
ax.legend(frameon=False)
ax.grid(alpha=.2)
fig.savefig(ROOT / 'figures/asian_control.png', dpi=180)
plt.close(fig)
print('Saved barrier monitoring and Asian control figures from recorded results.')
