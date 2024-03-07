import os

file_path = '/home/blueberry/cache/data/image_net_30/mapping.txt'


mapper = {}

with open(file_path) as file:
    content = file.readlines()
    for line in content:
        label, dir_name = line.strip(' \n').split(' ')
        dirs = os.listdir('/home/blueberry/cache/data/image_net_30/train')
        for key in dirs:
            if key == dir_name:
                mapper[key] = label


print(mapper)
print(len(mapper))



true_label = 0

with open('/home/blueberry/cache/data/image_net_30/mapper.txt', mode='w') as file:
    for dir_name, label in mapper.items():
        file.write(f'{dir_name} {true_label} {label}\n')
        true_label += 1