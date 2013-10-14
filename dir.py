
import os

cat_dir = "img/cat"
dog_dir = "img/dog"

for directory in [cat_dir, dog_dir]:
	for fn in os.listdir(directory):
		print directory, fn