import os
import json


train_images ="../train/images"
valid_images ="../valid/images"
test_images = "../test/images"

labels_train = "../train/labels"
labels_valid = "../valid/labels"
labes_test = "../test/labels"

output_train_file = "coco_train_annotations.json"
output_valid_file = "coco_valid_annotations.json"
output_test_file = "coco_test_annotations.json"

# Categories (from data.yaml)
categories = [{'id': 1, 'name': 'printer-gantry'}, {'id': 2, 'name': 'spaghetti'}]
