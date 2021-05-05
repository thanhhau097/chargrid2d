# Requirements
- Python 3.6.5
- Pytorch 1.2.0

# Training 
Note: because the data format is private, you should consider the function `def __convert_data(self, label_json):` in `https://github.com/thanhhau097/chargrid2d/blob/master/dataloader_utils/generate_mask.py#L27` based on your input format, as long as the outputs of this function is correct. (sorry for this mistake, I will update code for public dataset later)

To train model, first you need to run chargrid2d/dataloader_utils/generate_mask.py to generate data to train segmentation model from bounding box and text.

Then you need to run train.py file to train the model.

If you have any questions, please create an issue.

To train model, first you need to run chargrid2d/dataloader_utils/generate_mask.py to generate data to train segmentation model from bounding box and text.

To use with SROIE dataset, we need to do labeling corresponding class for each box in the image. You can read the code in `generate_masks.py` file, which will call this function:

```python
def __convert_data_sroie(self, label_json):
    std_out = []

    for item in label_json:
        for char in item['text']:
            if char not in self.__corpus__:
                self.__corpus__[char] = 1
            else:
                self.__corpus__[char] += 1

        std_out.append({
            'value': item['text'],
            'formal_key': item['class'],
            'key_type': 'value',
            'location': item['box']
        })

    return std_out
```

the input data in json file should be a list of dictionary with keys: `text`, `class`, `box`.

You could do the processing by using this (which was commented out in the main function):

```python
mask_generator = MaskGenerator()
mask_generator.process(lbl_fol, img_fol, out_fol)
mask_generator.generate_mask('.')
```

Then you need to run train.py file to train the model.

If you have any questions, please create an issue.
