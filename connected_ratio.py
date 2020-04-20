import sys
import dsynt.data

path = sys.argv[1]
data = dsynt.data.read_deep_conll(path)

n_connected = 0
n_rooted = 0
for sentence in data:
    c, rooted = dsynt.data.is_connected(sentence, details=True)
    if c:
        n_connected += 1
        if rooted:
            n_rooted += 1

print("N connected: %i / %i\t(%.2f)" % (n_connected, len(data), 100 * n_connected / len(data)))
print("N connected and rooted: %i / %i\t(%.2f)" % (n_rooted, len(data), 100 * n_rooted / len(data)))