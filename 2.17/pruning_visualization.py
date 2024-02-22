import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import func_helpers as fh


def pulse(t, s=1):
    return t/s *np.exp(-(t/s)**2)

shifts, endpoints = [-np.pi/6, np.pi/6], [-np.pi, np.pi]
NMAX, lengths = 500, (150, 350)

base_time = np.linspace(*endpoints, NMAX)
base_vecs = [pulse(base_time + s) for s in shifts]

print(base_time)
print(base_vecs)

# Generate unequally long timeseries from the base pulse
*idxs, = map(sorted, [np.random.randint(0, NMAX, l) for l in lengths])
unequal_pair = [(idx,vec[idx]) for vec, idx in zip(base_vecs, idxs)]



# Visualize whether the pruning worked ...
*pruned_pair, = reversed(fh.prune(*reversed(unequal_pair), ids=True))

fig, ((aa, ab), (ba, bb)) = plt.subplots(2,2, figsize=(14,10))

for vec in base_vecs:
    aa.scatter(base_time, vec)
aa.set_title("Two shifted pulses of equal length")

for item in unequal_pair:
    ab.scatter(*item)
ab.set_title("Two shifted pulses of with different # of samples")

for item in pruned_pair:
    bb.scatter(*item)
bb.set_title("Two pruned pulses with same # of samples")

ba.scatter(*list(zip(*pruned_pair))[1])
ba.set_aspect('equal', 'box')
ba.set_title("The area plot from the pruned pulse.")

plt.show()
plt.pause(0)



# cyclic

def xdy_ydx(pair):
    """Calculate the area integral between a pair of timeseries data."""
    x, y = ch.match_ends(normalize(mean_center(np.nan_to_num(np.asarray(pair)))))
    return x.dot(cyc_diff(y)) - y.dot(cyc_diff(x))


data = [[(fh.intify(f.timestamps),f.firingrates.tolist()) for f in \
        n.fields.values()] for n in neuron_data.values()]

ret = make_lead_matrix(flatten(data), intfunc=xdy_ydx)
LM, phases, perm, sortedLM, evals = ret


ts_names = list()
ts_names = ["--".join((nn, f)) for nn, nd in neuron_data.items() for f in \
            nd.fields.keys()]

# Function defined in plot_helpers.py
fig = plt.figure(figsize=(25,12))
idxs = plot_initial(ret, fig=fig)
fig


