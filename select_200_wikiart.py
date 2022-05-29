from email.errors import InvalidMultipartContentTransferEncodingDefect
import os
import random
from pathlib import Path

PATH = os.path.join("data", "train", "wikiart")

dest_sized_dataset = 200
img_counter = 0
img_paths = []

wikiart_folder = list(Path(PATH).glob('*.jpg'))

while(img_counter < dest_sized_dataset):

    index = random.randint(0, len(wikiart_folder))
    img_paths.append(str(wikiart_folder[index]))
    img_counter += 1

with open('wiki_art_selected.txt', 'w') as filehandle:
    for listitem in img_paths:
        filehandle.write('%s\n' % listitem)
