labels = ['water', 'cloudy', 'partly_cloudy', 'haze', 'selective_logging', 'agriculture', 'blooming', 'cultivation', 'habitation', 'road', 'bare_ground', 'clear', 'conventional_mine', 'artisinal_mine', 'slash_burn', 'primary', 'blow_down']
labels.sort()

with open('../rawInput/train.csv') as f, open('../rawInput/trained-processed.csv', 'w') as f2:
  f2.write('image_name,' + ','.join(labels) + '\n')
  f.readline()
  for l in f:
    filename, rawTags = l.strip().split(',')
    tags = rawTags.split(' ')
    f2.write(filename + ',' + ','.join(['1' if tag in tags else '0' for tag in labels]) + '\n')
