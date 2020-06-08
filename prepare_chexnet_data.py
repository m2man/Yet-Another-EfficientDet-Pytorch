import numpy as np
import json
import os
import random
import shutil
random.seed(1509)
os.chdir('/home/dxtien/dxtien_research/nmduy/Yet-Another-EfficientDet-Pytorch/')

#os.makedirs('pneumonia/annotations')
#os.makedirs('pneumonia/train')
#os.makedirs('pneumonia/val')

with open('/home/dxtien/dxtien_research/nmduy/data_covid_creation/data_pneu_bbox.json') as f:
    data_bbox = json.load(f)

DataFolder = '/home/dxtien/dxtien_research/COVID/CXR8/'

list_images = list(data_bbox.keys())
numb_images = len(list_images)
numb_images_each = int(numb_images/2)
train_portion = 0.8
numb_train_each = int(train_portion*numb_images_each)

abnormal_images = list_images[0:numb_images_each]
normal_images = list_images[numb_images_each:]

random.shuffle(abnormal_images)
random.shuffle(normal_images)

abnormal_images_train = abnormal_images[0:numb_train_each]

#### If only presence abnormal in the training data
#images_train = abnormal_images_train
####

#### both normal and abnormal in the training data
normal_images_train = normal_images[0:numb_train_each]
images_train = abnormal_images_train + normal_images_train
####

images_test = [x for x in list_images if x not in images_train]

# ======== TRAIN ANNOTATION =======

annotation = {}

# ===== INFO =====
field = 'info'
annotation[field] = {}
annotation[field]['desciption'] = "Annotation for CXR14 Bounding Box data in COCO format - Training"
annotation[field]['url'] = "Google Chest X-ray 14"
annotation[field]['version'] = "1.0"
annotation[field]['year'] = 2020
annotation[field]['contributor'] = "m2man"
annotation[field]['date_created'] = "2020-05-14"

# ===== LICENSES =====
field = 'licenses'
licenses_json = {}
licenses_json['id'] = 1
licenses_json['name'] = ""
licenses_json['url'] = ""
annotation[field] = [licenses_json]

# ===== CATEGORIES =====
field = 'categories'
categories_json = {}
categories_json['id'] = 1
categories_json['name'] = 'abnormal'
categories_json['supercategory'] = "None"
annotation[field] = [categories_json]

# ===== IMAGES =====
field = 'images'
list_images_json = []
for idx in range(len(images_train)):
    image_json = {}
    image_json['id'] = idx
    image_name = images_train[idx]
    image_name_short = image_name.split('/')[-1]
    image_json['file_name'] = image_name_short
    image_json['width'] = 1024
    image_json['height'] = 1024
    list_images_json.append(image_json)
annotation[field] = list_images_json

# ===== ANNOTATIONS =====
field = 'annotations'
list_annotations_json = []
idx_annotation = 0
for idx in range(len(images_train)):
    if len(data_bbox[images_train[idx]]['bbox']) > 0:
        annotations_json = {}
        annotations_json['id'] = idx_annotation
        annotations_json['image_id'] = idx
        annotations_json['category_id'] = 1
        annotations_json['iscrowd'] = 0
        bbox = data_bbox[images_train[idx]]['bbox']
        bbox = [int(x) for x in bbox]
        bbox[0] -= 1
        bbox[1] -= 1
        bbox[2] += 1
        bbox[3] += 1
        annotations_json['area'] = bbox[2] * bbox[3]
        annotations_json['bbox'] = bbox
        annotations_json['segmentation'] = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]]]
        list_annotations_json.append(annotations_json)
        idx_annotation += 1
annotation[field] = list_annotations_json
annotation_train = annotation
del annotation

with open('datasets/pneumonia_with_none/annotations/instances_train.json', 'w') as f:
    json.dump(annotation_train, f)



# ======== TEST ANNOTATION =======

annotation = {}

# ===== INFO =====
field = 'info'
annotation[field] = {}
annotation[field]['desciption'] = "Annotation for CXR14 Bounding Box data in COCO format - Test"
annotation[field]['url'] = "Google Chest X-ray 14"
annotation[field]['version'] = "1.0"
annotation[field]['year'] = 2020
annotation[field]['contributor'] = "m2man"
annotation[field]['date_created'] = "2020-05-14"

# ===== LICENSES =====
field = 'licenses'
licenses_json = {}
licenses_json['id'] = 1
licenses_json['name'] = ""
licenses_json['url'] = ""
annotation[field] = [licenses_json]

# ===== CATEGORIES =====
field = 'categories'
categories_json = {}
categories_json['id'] = 1
categories_json['name'] = 'abnormal'
categories_json['supercategory'] = "None"
annotation[field] = [categories_json]

# ===== IMAGES =====
field = 'images'
list_images_json = []
for idx in range(len(images_test)):
    image_json = {}
    image_json['id'] = idx
    image_name = images_test[idx]
    image_name_short = image_name.split('/')[-1]
    image_json['file_name'] = image_name_short
    image_json['width'] = 1024
    image_json['height'] = 1024
    list_images_json.append(image_json)
annotation[field] = list_images_json

# ===== ANNOTATIONS =====
field = 'annotations'
list_annotations_json = []
idx_annotation = 0
for idx in range(len(images_test)):
    if len(data_bbox[images_test[idx]]['bbox']) > 0:
        annotations_json = {}
        annotations_json['id'] = idx_annotation
        annotations_json['image_id'] = idx
        annotations_json['category_id'] = 1
        annotations_json['iscrowd'] = 0
        bbox = data_bbox[images_test[idx]]['bbox']
        bbox = [int(x) for x in bbox]
        bbox[0] -= 1
        bbox[1] -= 1
        bbox[2] += 1
        bbox[3] += 1
        annotations_json['area'] = bbox[2] * bbox[3]
        annotations_json['bbox'] = bbox
        annotations_json['segmentation'] = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3]]]
        list_annotations_json.append(annotations_json)
        idx_annotation += 1
annotation[field] = list_annotations_json
annotation_test = annotation
del annotation

with open('datasets/pneumonia_with_none/annotations/instances_val.json', 'w') as f:
    json.dump(annotation_test, f)


# =========== COPY IMAGES ===========
for img in images_train:
    img_short = img.split('/')[-1]
    shutil.copy2(DataFolder+img, f'datasets/pneumonia_with_none/train/{img_short}')

for img in images_test:
    img_short = img.split('/')[-1]
    shutil.copy2(DataFolder+img, f'datasets/pneumonia_with_none/val/{img_short}')
