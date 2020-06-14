import logging
import numpy as np
import os
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.utils.logger import setup_logger
from collections import namedtuple

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)

# Following approach from Cityscapes Scripts: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


# TODO: label ids are likely incorrect.
labels = [
    #       name                  id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'tree'               ,  0 ,      0 , 'nature'          , 0       , False        , False        , (  0,  0,  0) ),
    Label(  'grass'              ,  1 ,      1 , 'nature'          , 0       , False        , True         , (111, 74,  0) ), # TODO: assuming that there is typo in "gras"
    Label(  'other vegetation'   ,  2 ,      2 , 'nature'          , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'dirt'               ,  3 ,      3 , 'flat'            , 1       , False        , True         , (128, 64,128) ),
    Label(  'gravel'             ,  4 ,      4 , 'flat'            , 1       , False        , True         , (244, 35,232) ),
    Label(  'rocks'              ,  5 ,      5 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'water'              ,  6 ,      6 , 'flat'            , 1       , False        , False        , (230,150,140) ),
    Label(  'paved area'         ,  7 ,      7 , 'construction'    , 2       , False        , True         , ( 70, 70, 70) ),
    Label(  'pool'               ,  8 ,      8 , 'construction'    , 2       , False        , True         , (102,102,156) ),
    Label(  'person'             ,  9 ,      9 , 'human'           , 3       , True         , False        , (220, 20, 60) ),
    Label(  'dog'                ,  10 ,    10 , 'animal'          , 4       , True         , False        , (255,  0,  0) ),
    Label(  'car'                ,  11 ,    11 , 'vehicle'         , 5       , True         , False        , (  0,  0,142) ),
    Label(  'bicycle'            ,  12 ,    12 , 'vehicle'         , 5       , True         , False        , (  0,  0, 70) ),
    Label(  'roof'               ,  13 ,    13 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'wall'               ,  14 ,    14 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'fence'              ,  15 ,    15 , 'object'          , 6       , False        , True         , (153,153,153) ),
    Label(  'fence-pole'         ,  16 ,    16 , 'object'          , 6       , False        , True         , (153,153,150) ),
    Label(  'window'             ,  17 ,    17 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'door'               ,  18 ,    18 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'obstacle'           ,  19 ,    19 , 'object'          , 6       , False        , False        , (220,220,  0) ),
]


def get_drone_files(image_dir, gt_dir):

    from os import listdir
    from os.path import isfile, join

    image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    # get corresponding GT file
    gt_files = []
    for file in image_files:
        filename = os.path.splitext(file)[0]
        gt_files.append(filename+".png")

    # assert len(image_files) == len(gt_files), f"Number of files in image \
    #     directory {len(image_files)} is different from number of GT \
    #     files {len(gt_files)}"

    # add file path and combine image and gt files into list of tuples
    combined_files = []
    for image, gt in zip(image_files, gt_files):
        combined_files.append(
            (
                os.path.join(image_dir, image),
                os.path.join(gt_dir, gt)
            )
        )

    return combined_files


def load_drone_semantic(image_dir, gt_dir):

    results = []

    for image_file, gt_file in get_drone_files(image_dir, gt_dir):

        # PIL Image is efficient as it doesn't read image to retrieve size
        im = Image.open(image_file)
        width, height = im.size

        results.append({
            "file_name": image_file,
            "sem_seg_file_name": gt_file,
            "height": height,
            "width": width
            }
        )

    assert len(results),  f"No images found in {image_dir}!"
    assert PathManager.isfile(results[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa

    return results


if __name__ == "__main__":
    """
    Test the drone dataset loader.

    Usage:
        python drone_dataset_processing.py \
            semantic_drone_dataset/original_images/ semantic_drone_dataset/label_images_semantic/
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("gt_dir")
    args = parser.parse_args()

    logger = setup_logger(name=__name__)

    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)

    dirname = "drone-data-vis"
    os.makedirs(dirname, exist_ok=True)

    dicts = load_drone_semantic(args.image_dir, args.gt_dir)
    logger.info("Done loading {} samples.".format(len(dicts)))

    stuff_colors = [k.color for k in labels if k.trainId != 255]
    stuff_classes = [k.name for k in labels if k.trainId != 255]
    meta = Metadata().set(stuff_colors=stuff_colors, stuff_classes=stuff_classes)

    for d in dicts:
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)

