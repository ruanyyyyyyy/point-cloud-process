
# Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 20.04)
- Python 3.7
- PyTorch 1.8.1
- cuda 11.2

Credit to AlexGeControl https://github.com/AlexGeControl/3D-Point-Cloud-Analytics/tree/master/workspace/assignments/project-01-kitti-detection-pipeline
This repo refers lots of scripts in the above link.

# Build Object Classification Dataset from KITTI 3D Object

## KITTI dataset
Refer to the paper about recording platform, sensor setup, structure of the provided zip-files, development kit, object coordinates and transformations among different coordinates. The paper provides detailed explanations about those information.
http://www.cvlibs.net/publications/Geiger2013IJRR.pdf


## Extract object point cloud from KITTI whole road point clouds

```
python scripts/extract.py -i ./data/KITTI/object/training/ -o ./data/processed_KITTI
```

#### Algorithm Workflow
(copy from https://raw.githubusercontent.com/AlexGeControl/3D-Point-Cloud-Analytics/master/workspace/assignments/project-01-kitti-detection-pipeline/README.md)
The algorithm workflow in pseudo code is as follows. For each frame(velodyne-image-calib-label tuple) from KITTI 3D Object:

* First, perform **ground plane & object segmentation** on **Velodyne measurements**.
    * The implementation from [/workspace/assignments/04-model-fitting/clustering.py](assignment 04 of model fitting) is used here for region proposal 
* Build **radius-nearest neighbor** search tree upon segmented objects for later **label association**
* Then load label and associate its info with segmented objects as follows:
    * First, map **object center** from **camera frame** to **velodyne frame** using the parameters from corresponding calib file.
    * Query the search tree and identify a **coarse ROI**. 
        * The radius is determined using the dimensions of the bounding box. Euclidean transform is isometry.
    * Map the point measurements within the sphere into **object frame**.
    * Identify a **refined ROI** through **object bounding box filtering**.
    * Perform **non-maximum suppression on segmented object ids** for final label association:
        * Perform an ID-count on points inside the bounding box.
        * Identify the segmented object with maximum number of measurements inside the bounding box.
        * Associate the KITTI 3D Object label with the segmented object of maximum ID-count.
* Finally, write the **point cloud with normal** and corresponding **metadata** info persistent storage. 

## Dataset Analytics before Deep Learning Modelling

The following command would visualize Class Distribution in pie chart (`plt.pie`)and show Influence of Distance on Measurement Density in line chart(`sns.lineplot`). They are saved in `output/analyze_dataset`

```
python scripts/generating-training-set.py -i ./data/processed_KITTI -m analyze
```

1. Before training the network, data quality check must be performed to minimize the effect of uneven class distribution, etc, on model building.

From the above distribution image, we know:
significant uneven class distribution exists.  `cyclist` and `pedestrian` is much less than `misc` and `vehicle`.
We should use data augmentation(rotation around z axis here) to get more data compared with percentage of `misc` (line292-line325).

2. For efficient deep-learning network training, all input point clouds should be transformed to the same size. However, the number and density of lidar measurements is influenced by the object's distance to ego vehicle. So measurement count analytics must be performed before choosing FoV and input size.

From the above images, we know:
Measurement Density means how many points in an object. From the line chart we can choose a threshold, 25 meters. Objects in area >25 meters have too less points so we ignore them. We define the ROI as area <=25 meters away from ego lidar.
Also for batch training, we resample each object in FOV to 64 points according to y axis(num. meansurements) in that line chart.


# Object Classification Network
PointNet++ with cross entropy loss is used as object classification network

Use pointnet++ implementation in https://github.com/erikwijmans/Pointnet2_PyTorch.git

## Install Pointnet++ library

Install them directly (this can also be used in a requirements.txt)
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
Then, you can use pointnet2 by importing `from models.pointnet2_msg_cls import PointNet2ClassificationMSG` 

An example:
```
from models.pointnet2_msg_cls import PointNet2ClassificationMSG

model = PointNet2ClassificationMSG()
```

## Training object classification network

Note that use normal info in object classification. So shape of input point cloud is (bs, 64, 6)
Run the following commands to train the network on the resampled dataset:

```bash
python pointnet2/train.py --ckpt_save_interval 2 --epochs 20
```

After 20 epochs the network can achieve 0.92 accuracy on validation dataset

Resume from previous ckpt
```
python pointnet2/train.py --ckpt_save_interval 2 --epochs 40 --resume true --ckpt output/default/ckpt/checkpoint_epoch_20.pth --lr 0.00016
```

Visualizae training loss on Tensorboard. After this the traning loss can be monitored inside local browser.

```
tensorboard --logdir=runs
```

Generate confusion matrix and the classification report from sklearn
```
python pointnet2/train.py --mode test --ckpt output/default/ckpt/checkpoint_epoch_20.pth
```

Corresponding code in train.py

```
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

y_truth = np.hstack(y_truths)
y_pred = np.hstack(y_preds)

# get confusion matrix
conf_mat = confusion_matrix(y_truth, y_pred)
# change to percentage:
conf_mat = np.dot(
    np.diag(1.0 / conf_mat.sum(axis=1)),
    conf_mat
)
conf_mat = 100.0 * conf_mat


plt.figure(figsize = (10, 10))
sn.heatmap(conf_mat, annot=True, xticklabels=labels, yticklabels=labels)
plt.title('KITTI 3D Object Classification -- Confusion Matrix')
plt.savefig('./output/confusion_matrix.png')
# plt.show()

log_print(
    classification_report(
        y_truth, y_pred, 
        target_names=labels
    ), log_f
)    

```

Confusion matrix png is stored in `./output/confusion_matrix.png`. The classification report is stored in `./output/default/test/log.txt`

# Detection pipeline

Use already trained classification network to do detection with KITTI velodyne and calib folder.

```
python scripts/detect.py -i ./data -c output/default/ckpt/checkpoint_epoch_20.pth
```
The result txt would be saved in `./output/result_KITTI`

#### Algorithm Workflow
(totally copy from https://raw.githubusercontent.com/AlexGeControl/3D-Point-Cloud-Analytics/master/workspace/assignments/project-01-kitti-detection-pipeline/README.md)
With the object classification network, the final object detection pipeline can be set up as follows:

* First, perform **ground plane & object segmentation** on **Velodyne measurements**.
* For each segmented **foreground object**, **preprocess** it according to **classification network input specification**. [click here](pointnet++/detect.py)
    * Filter out objects with too few measurements;
    * Filter object that is too far away from ego vehicle;
    * Resample object point cloud according to classification network input specification;
    * Substract mean to make the point cloud zero-centered.
* Run **batch prediction** on the above resampled point clouds and get **object category and prediction confidence**.
* Fit the cuboid using **Open3D axis aligned bounding box** in **Velodyne frame**, then transform to **camera frame** for **KITTI evaluation output**.

The corresponding Python implementation is shown below:

```python
    # 0. generate I/O paths:
    input_velodyne = os.path.join(dataset_dir, 'velodyne', f'{index:06d}.bin')
    input_params = os.path.join(dataset_dir, 'calib', f'{index:06d}.txt')
    output_label = os.path.join(dataset_dir, 'shenlan_pipeline_pred_2', 'data', f'{index:06d}.txt')

    # 1. read Velodyne measurements and calib params:
    point_cloud = measurement.read_measurements(input_velodyne)
    param = measurement.read_calib(input_params)

    # 2. segment ground and surrounding objects -- here discard intensity channel:
    segmented_ground, segmented_objects, object_ids = segmentation.segment_ground_and_objects(point_cloud[:, 0:3])

    # 3. predict object category using classification network:
    config = {
        # preprocess:
        'max_radius_distance': max_radius_distance,
        'num_sample_points': num_sample_points,
        # predict:
        'msg' : True,
        'batch_size' : 16,
        'num_classes' : 4,
        'batch_normalization' : False,
        'checkpoint_path' : 'logs/msg_1/model/weights.ckpt',
    }
    model = load_model(config)
    predictions = predict(segmented_objects, object_ids, model, config)
    
    # TODO: refactor decoder implementation
    decoder = KITTIPCDClassificationDataset(input_dir='/workspace/data/kitti_3d_object_classification_normal_resampled').get_decoder()

    # debug mode:
    if (debug_mode):
        # print detection results:
        for class_id in predictions:
            # show category:
            print(f'[{decoder[class_id]}]')
            # show instances:
            for object_id in predictions[class_id]:
                print(f'\t[Object ID]: {object_id}, confidence {predictions[class_id][object_id]:.2f}')

        # visualize:
        bounding_boxes = visualization.get_bounding_boxes(
            segmented_objects, object_ids, 
            predictions, decoder
        )
        o3d.visualization.draw_geometries(
            [segmented_ground, segmented_objects] + bounding_boxes
        )
    
    # 4. format output for KITTI offline evaluation tool:
    label = output.to_kitti_eval_format(
        segmented_objects, object_ids, param,
        predictions, decoder
    )
    label.to_csv(output_label, sep=' ', header=False, index=False)
```

## Evaluation

Use this kitti eval tool: https://github.com/prclibo/kitti_eval/blob/955c04c4afc4cbc85638fb1e7b237068415c1d05/evaluate_object_3d_offline.cpp
```
 ./evaluate_object_3d_offline /local-scratch/yuer/projects/pc_asgn/finalproj/data/KITTI/object/training/label_2 /local-scratch/yuer/projects/pc_asgn/finalproj/output/final_results
```
Note that make sure `final_results` has a subdirectory named `data` where store all the txt files.
The eval results would also be put in `final_results`, including `plot` folder and those statistics txt.