# Requirements
- Python 3.6.5
- Pytorch 1.2.0

# Training 
Note: because the data format is private, you should consider the function `def __convert_data(self, label_json):` in `https://github.com/thanhhau097/chargrid2d/blob/master/dataloader_utils/generate_mask.py#L27` based on your input format, as long as the outputs of this function is correct. (sorry for this mistake, I will update code for public dataset later)

To train model, first you need to run chargrid2d/dataloader_utils/generate_mask.py to generate data to train segmentation model from bounding box and text.

Then you need to run train.py file to train the model.

If you have any questions, please create an issue.

