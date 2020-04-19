# This is default implementation of inference_handler: 
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py
# SM specs: https://sagemaker.readthedocs.io/en/stable/using_pytorch.html


# TODO list
# 1. add support of multi-GPU instances - if GPU devices > 1, do round robin
# 2. do we need to support checkpoints (optimizers, LR etc.)


from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2.data.transforms as T
#from detectron2.engine import DefaultPredictor
import torch
import numpy as np
import cv2 # TODO: delete

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
from sagemaker.content_types import CONTENT_TYPE_JSON, CONTENT_TYPE_CSV, CONTENT_TYPE_NPY # TODO: for local debug only. Remove or comment when deploying remotely.
from six import StringIO, BytesIO  # TODO: for local debug only. Remove or comment when deploying remotely.
import os


class CocoPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """ 
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def model_fn(model_dir):
    """
    Deserialize and load D2 model. This method is called automatically by Sagemaker.
    model_dir is location where your trained model will be downloaded.
    """
    
    # restoring trained model
    print(model_dir)
    config_file = os.path.join(model_dir, "config.yaml")
    model_path = os.path.join(model_dir, "model_final.pth")
    
    # based on this doc: https://detectron2.readthedocs.io/tutorials/models.html
    cfg = get_cfg()
    cfg.merge_from_file(config_file) # TODO: since we already have a full fledge config, do we need to do merge_from_file?
    cfg.MODEL.WEIGHTS = model_path
    pred = CocoPredictor(cfg)
    
    return pred

def input_fn(input_data, content_type):
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO implement here some sort of round robin
    np_array = decoder.decode(input_data, content_type) # TODO: uncomment
#    tensor = torch.FloatTensor(np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array) #TODO: uncomment
    print("DEBUG: np_array", np_array)
    return np_array

def predict_fn(inputs, pred):
    # accroding to D2 rquirements: https://detectron2.readthedocs.io/tutorials/models.html
    outputs = pred(inputs)
    return outputs

# Uncomment if you need custom preprocessing of predictor output before sending to the client.
# By default, this one will be used: https://github.com/aws/sagemaker-pytorch-serving-container/blob/63dfd491ee50539b8c787088672a683ed7df03b3/src/sagemaker_pytorch_serving_container/default_inference_handler.py#L93-L108
# def output_fn(prediction, response_content_type):
#     pass


def _npy_serialize(data):
    """
    Args:
        data:
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


if __name__ == "__main__":
    
    # Test Stub to replicate sequence of calls at inference endpoint. Keep it for local debugging. 
    # This code won't be executed on the Sagemaker endpoint
    
    # 1. Get the image. Make sure that you point to your valid dir with COCO2017 val dataset
    data_dir = "../datasets/coco/"
    dataset  = "val2017"
    annFile='{}{}/annotations/instances_{}.json'.format(data_dir,dataset, dataset)
    coco=COCO(annFile)
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
    imgIds = coco.getImgIds(catIds=catIds );
    imgIds = coco.getImgIds(imgIds = [324158])
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    I = io.imread(img['coco_url'])    
    I_chw = np.transpose(I,(2,0,1)) # convert to D2 expected format CHW
    
    # 2. Serialize the data
    _npy_serialize(I_chw)
    
    # 3. Deserialize the data
    image = input_fn(I_chw,CONTENT_TYPE_NPY)

    # 4. 
    pred = model_fn("../model2_dir") #TODO sending dummy var
    outputs = predict_fn(image, pred)
    
    print(outputs)
    

