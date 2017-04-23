import numpy as np
np.set_printoptions(linewidth=200)

labels = set()

#labels
with open('../rawInput/train.csv') as f:
	f.readline()
	for l in f:
		tags = l.strip().split(',')[1].split(' ')
		for tag in tags:
			labels.add(tag)

print('labels', labels, len(labels))

# correlations
labels = [l for l in labels]
print('labels', labels)
mapping = {l: i for (i, l) in enumerate(labels)}
print('mapping', mapping)
combinations = {}
maxTagsNumber = 0
correlations = [[0 for i in mapping] for j in mapping]

with open('../rawInput/train.csv') as f:
	f.readline()
	for l in f:
		tags = l.strip().split(',')[1].split(' ')
		maxTagsNumber = max(maxTagsNumber, len(tags))
		for tag in tags:
			for otherTag in tags:
				correlations[mapping[tag]][mapping[otherTag]] += 1
		tags.sort()
		combination = '-'.join(tags)
		combinations[combination] = combinations[combination] if combination in combinations else 0
		combinations[combination] += 1

combinations = [(name, combinations[name]) for name in combinations]
combinations.sort(key=lambda x: -x[1])

print('correlations')
print(labels)
print(np.matrix(correlations))

with open('label-correlations.csv', 'w') as f:
	f.write(',')
	f.write(','.join(labels) + '\n')
	for i, label in enumerate(labels):
		f.write(','.join([label] + [str(k) for k in correlations[i]]) + '\n')

print('combinations', combinations)
print('maxTagsNumber', maxTagsNumber)
