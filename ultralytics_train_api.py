# pip install ultralytics
from ultralytics import YOLO, checks, hub
checks()


hub.login('1a4016486375ce3e30d5d64623136281102d96afff')

model = YOLO('https://hub.ultralytics.com/models/UZ7wmNgLKE9e03vzK55J')
results = model.train()