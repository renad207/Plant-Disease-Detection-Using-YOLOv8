# import libraries



from roboflow import Roboflow
import os, shutil
from collections import Counter
import os
import random
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import cv2
import seaborn as sns



#Download Dataset

rf = Roboflow(api_key="NaGZDBPtaV5Ak4QgG7sf")
project = rf.workspace("fyp-advun").project("crop-disease-2rilx")
version = project.version(1)
dataset = version.download("yolov8")

#Merge Dataset



def merge_data(base_path, out_img, out_lbl):
    splits = ["train", "valid", "test"]

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    for split in splits:
        img_dir = os.path.join(base_path, split, "images")
        lbl_dir = os.path.join(base_path, split, "labels")

        for img in os.listdir(img_dir):
            new_name = f"{split}_{img}"

            shutil.copy(os.path.join(img_dir, img),
                        os.path.join(out_img, new_name))

            label_name = os.path.splitext(img)[0] + ".txt"

            if os.path.exists(os.path.join(lbl_dir, label_name)):
                shutil.copy(os.path.join(lbl_dir, label_name),
                            os.path.join(out_lbl,
                            new_name.replace(".jpg", ".txt")))
                
# Enable merging

merge_data(
    dataset.location,
    "/content/all_images",
    "/content/all_labels"
)

#Split 60/20/20



def split_dataset(images_path, labels_path, output_path):
    random.seed(42)

    images = [img for img in os.listdir(images_path)
              if img.endswith(('.jpg', '.png'))]

    random.shuffle(images)

    train_end = int(0.6 * len(images))
    val_end = int(0.8 * len(images))

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)

        for img in splits[split]:
            shutil.copy(os.path.join(images_path, img),
                        f"{output_path}/{split}/images/{img}")

            label = os.path.splitext(img)[0] + ".txt"

            if os.path.exists(os.path.join(labels_path, label)):
                shutil.copy(os.path.join(labels_path, label),
                            f"{output_path}/{split}/labels/{label}")
                
# Enable splitting

split_dataset(
    "/content/all_images",
    "/content/all_labels",
    "/content/dataset_final"
)



def count_classes(labels_path):
    counter = Counter()

    for file in os.listdir(labels_path):
        if not file.endswith(".txt"):
            continue

        with open(os.path.join(labels_path, file)) as f:
            lines = f.readlines()

        for line in lines:
            class_id = int(line.split()[0])
            counter[class_id] += 1

    return counter


# apply only on the train subset
counts = count_classes("/content/dataset_final/train/labels")

print("Class Distribution:")
for k, v in counts.items():
    print(f"class {k}: {v}")
    

#Checking of data leakage


train_files = set(os.listdir("/content/dataset_final/train/images"))
test_files = set(os.listdir("/content/dataset_final/test/images"))

common = train_files.intersection(test_files)

print("Common files:", len(common))



images_path = "/content/dataset_final/train/images"
labels_path = "/content/dataset_final/train/labels"

missing_labels = []
missing_images = []

# check images -> labels
for img in os.listdir(images_path):
    label = os.path.splitext(img)[0] + ".txt"
    if not os.path.exists(os.path.join(labels_path, label)):
        missing_labels.append(img)

# check labels -> images
for lbl in os.listdir(labels_path):
    img = os.path.splitext(lbl)[0] + ".jpg"
    if not os.path.exists(os.path.join(images_path, img)):
        missing_images.append(lbl)

print("Images without labels:", len(missing_labels))
print("Labels without images:", len(missing_images))

#  number of images before balance



train_images = "/content/dataset_final/train/images"

print("Before balance:", len(os.listdir(train_images)))

#Balance Function

def duplicate_minority(images_path, labels_path, target_class=0, factor=2):
    images = os.listdir(images_path)

    for img in images:
        label_file = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            if int(line.split()[0]) == target_class:

                for i in range(factor):
                    new_name = f"dup_{i}_{img}"

                    shutil.copy(os.path.join(images_path, img),
                                os.path.join(images_path, new_name))

                    shutil.copy(label_path,
                                os.path.join(labels_path,
                                new_name.replace(".jpg",".txt")))
                    
##Class Balance Plan

balance_plan = {
    19:2, 15:2, 22:2, 13:2, 14:2, 21:2,
    10:2, 20:2, 6:2, 16:2, 17:2,
    5:2, 1:2, 11:2, 2:2, 9:2, 7:2, 8:2
}

#Apply Balance (TRAIN ONLY)

for cls, factor in balance_plan.items():
    duplicate_minority(
        "/content/dataset_final/train/images",
        "/content/dataset_final/train/labels",
        target_class=cls,
        factor=factor
    )

print("Balance Done ")

#  number of images after balance



train_images = "/content/dataset_final/train/images"

print("After balance:", len(os.listdir(train_images)))


# count images in every class and print the number

counter = Counter()

for file in os.listdir("/content/dataset_final/train/labels"):
    if file.endswith(".txt"):
        with open(os.path.join("/content/dataset_final/train/labels", file)) as f:
            for line in f:
                cls = int(line.split()[0])
                counter[cls] += 1

for k, v in sorted(counter.items()):
    print(f"class {k}: {v}")
    

#print classes names

with open(dataset.location + "/data.yaml") as f:
    original_yaml = yaml.safe_load(f)

print(original_yaml["names"])


data_yaml = {
    "path": "/content/dataset_final",
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",

    "names": [
        'Chili___Anthracnose_fruit',
        'Chili___Bacterial_leaf_spot',
        'Chili___Healthy_fruit',
        'Chili___Healthy_leaf',
        'Chili___Mosaic_virus_leaf',
        'Eggplant___Cercospora_leaf_spot',
        'Eggplant___Colorado_potato_beetle',
        'Eggplant___Fruit_rot',
        'Eggplant___Healthy_fruit',
        'Eggplant___Healthy_leaf',
        'Potato___Alternaria_solani_leaf',
        'Potato___Common_scab_fruit',
        'Potato___Healthy_fruit',
        'Potato___Healthy_leaf',
        'Potato___Phytopthora_infestans_leaf',
        'Tomato___Antrhacnose_fruit',
        'Tomato___Bacterial_spot_leaf',
        'Tomato___Early_blight_leaf',
        'Tomato___Healthy_fruit',
        'Tomato___Healthy_leaf',
        'Tomato___Late_blight_leaf',
        'Tomato___Leaf_mold',
        'Tomato___Tomato_yellow_leaf_curl_virus'
    ]
}

with open("/content/dataset_final/data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("data.yaml")


# model train

model = YOLO("yolov8s.pt")

model.train(
    data="/content/dataset_final/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,

    device=0,

    optimizer="AdamW",
    lr0=0.001,

    patience=10,

)

 # train model on augmentes data
 


# Augmentation Settings
aug_params = {
    "fliplr": 0.0,
    "flipud": 0.0,

    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,

    "degrees": 10,
    "scale": 0.5,
    "translate": 0.1,

    "mosaic": 1.0
}

model = YOLO("yolov8s.pt")

# experiment / training
results = model.train(
    data="/content/dataset_final/data.yaml",
    epochs=50,
    imgsz=320,
    batch=16,

    **aug_params,

    project="augmentation_experiments",
    name="exp_no_flip",
    save=True
)

print("Experiment finished")




#Evaluation Matrix 


# load trained model
model = YOLO("/content/best (4).pt")

# validation
metrics = model.val()

print("===== METRICS =====")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

print("\n Confusion Matrix saved in runs folder")





#test the model on external images

# Load model
model = YOLO("/content/best (4).pt")

# Predict on external test images
model.predict(
    source="/content/test_img",
    save=True,
    conf=0.25
)

# Show prediction results
predict_path = "/content/runs/detect/predict"

for img_name in os.listdir(predict_path):

    if img_name.endswith(('.jpg', '.png', '.jpeg')):

        img_path = os.path.join(predict_path, img_name)

        image = Image.open(img_path)

        plt.figure(figsize=(10,8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(img_name)
        plt.show()

# Evaluation Metrics
results = model.val(data="/content/dataset_final/data.yaml")

print("\nEvaluation Metrics:\n")

print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")





#test the model that traind on augmented dataset on external images
# Load model
model = YOLO("/content/best (5).pt")

# Predict on external test images
model.predict(
    source="/content/test_img",
    save=True,
    conf=0.25
)

# Show prediction results
predict_path = "/content/runs/detect/predict"

for img_name in os.listdir(predict_path):

    if img_name.endswith(('.jpg', '.png', '.jpeg')):

        img_path = os.path.join(predict_path, img_name)

        image = Image.open(img_path)

        plt.figure(figsize=(10,8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(img_name)
        plt.show()

# Evaluation Metrics
results = model.val(data="/content/dataset_final/data.yaml")

print("\nEvaluation Metrics:\n")

print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
print(f"mAP@0.5: {results.box.map50:.4f}")



# Severity Estimation

model = YOLO("/content/best (5).pt")

folder_path = "/content/dataset_final/test/images"

output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

data = []

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)

    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w, _ = image.shape
    image_area = h * w

    results = model(image)
    boxes = results[0].boxes

    count = len(boxes)
    total_disease_area = 0

    if count == 0:
        level = "No Disease"
    else:
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            label_name = model.names[cls]

            if "Healthy" in label_name:
                continue

            box_area = (x2 - x1) * (y2 - y1)
            total_disease_area += box_area

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = float(box.conf[0])

            label = f"{label_name} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if total_disease_area == 0:
            level = "No Disease"
        else:
            severity = total_disease_area / image_area

            if severity < 0.1:
                level = "Mild"
            elif severity < 0.3:
                level = "Moderate"
            else:
                level = "Severe"

    cv2.putText(image, f"Count: {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"Severity: {level}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, image)

    data.append([img_name, count, level])

df = pd.DataFrame(data, columns=["Image", "Count", "Severity"])

print("\nResults Table:\n")
print(df)

df.to_csv("results.csv", index=False)




# Counts Estimation


results_folder = "results"

no_disease = df[df["Severity"] == "No Disease"].head(2)
mild = df[df["Severity"] == "Mild"].tail(1)
moderate = df[df["Severity"] == "Moderate"].tail(1)
severe = df[df["Severity"] == "Severe"].tail(1)
selected = pd.concat([no_disease, mild, moderate , severe])

plt.figure(figsize=(10, 6))

for i, (_, row) in enumerate(selected.iterrows()):
    img_name = row["Image"]
    severity = row["Severity"]

    img_path = os.path.join(results_folder, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print("Error loading:", img_name)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(severity)
    plt.axis("off")

plt.tight_layout()
plt.show()



#Visualization

#bar chart
severity_counts = df["Severity"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(severity_counts.index, severity_counts.values)

plt.title("Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Images")

plt.show()


#Pie Chart
plt.figure(figsize=(6,6))

plt.pie(
    severity_counts.values,
    labels=severity_counts.index,
    autopct='%1.1f%%'
)

plt.title("Severity Percentage Distribution")
plt.show()


#boxplot



plt.figure(figsize=(6,4))
sns.boxplot(x="Severity", y="Count", data=df)

plt.title("Object Count per Severity Level")
plt.show()






