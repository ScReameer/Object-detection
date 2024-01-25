import os
import xml.etree.ElementTree as ET
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import io
from sklearn.model_selection import StratifiedShuffleSplit

class CustomImageDataset(Dataset):
    def __init__(
        self, 
        data_path,
        label_encoder,
        input_size=None,
        split=None, 
        random_state=42,
        test_size=0.3,
        transform=None,
        device='cpu'
        ):
        """Dataset for image detection

        Args:
            `data_path` (`str`): path to data
            `split` ([`None`, `str`], optional): Splits data to `train`, `valid` and `test`. Defaults to `None` (full dataset without split).
            `label_encoder` (sklearn.preprocessing.LabelEncoder): Fitted label encoder to encode classes
            `target_size` (`tuple`): network input size of images to resize bounding boxes
            `random_state` (`float`, optional): Random state for group splitting. Defaults to `42`
            `test_size` (`float`, optional): Size of test data, when `split` is not `None`. Defaults to `0.2`
            `transform` (`torchvision.transforms.v2.Compose`, optional): Augmentation and/or normalization. Defaults to `None`.
            `device` (`str`, optional): Move data to `cuda` or `cpu`. Defaults to `cpu`.
        
        """
        super().__init__()
        self.data_path_annotations = data_path + 'Annotations'
        self.data_path_images = data_path + 'JPEGImages'
        self.split = split
        self.transform = transform
        self.input_size = input_size
        self.device = device
        self.label_encoder = label_encoder
        self.items = []
        # Parse xml annotations
        for xml_file in os.listdir(self.data_path_annotations):
            tree = ET.parse(os.path.join(self.data_path_annotations, xml_file))
            root = tree.getroot()
            size = tree.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            for obj in root.findall('object'):
                bbx = obj.find('bndbox')
                xmin = float(bbx.find('xmin').text)
                ymin = float(bbx.find('ymin').text)
                xmax = float(bbx.find('xmax').text)
                ymax = float(bbx.find('ymax').text)
                # Encode label
                label = self.label_encoder.transform([obj.find('name').text])[0]
                result = (
                    root.find('filename').text,
                    label,
                    width,
                    height,
                    xmin,
                    ymin,
                    xmax,
                    ymax
                )
                self.items.append(result)
        
        # Split data to train, valid and test samples
        if split:
            self.random_state = random_state
            self.test_size = test_size
            self.sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            # Get train indexes
            train_idxs, remainder_idxs = list(self.sss.split(
                [item[0] for item in self.items], 
                [item[1] for item in self.items]
            ))[0]
            # Get valid and test indexes (1:1)
            half_total_remainder_len = len(remainder_idxs) // 2
            valid_idxs, test_idxs = remainder_idxs[:half_total_remainder_len], remainder_idxs[half_total_remainder_len:]
            # Split data
            match split:
                case 'train':
                    self.items = [self.items[train_idx] for train_idx in train_idxs]
                case 'valid':
                    self.items = [self.items[valid_idx] for valid_idx in valid_idxs]
                case 'test':
                    self.items = [self.items[test_idx] for test_idx in test_idxs]
        # Make dataframe for easier override __getitem__
        self.df = pd.DataFrame(
            self.items, 
            columns=['name', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
        )
        self.unique_images = self.df['name'].unique()
        
    def __len__(self):
        return len(self.unique_images)
    
    def __getitem__(self, idx):
        # Get image name, convert to tensor
        image_name = self.unique_images[idx]
        img = io.read_image(os.path.join(self.data_path_images, image_name)).to(self.device)
        # Get original shape of image
        orig_img_shape_y, orig_img_shape_x = img.shape[1], img.shape[2]
        # Apply transforms to image
        if self.transform:
            img = self.transform(img)
        # Get all rows with one unique image [idx]
        all_rows = self.df[self.df['name'] == image_name]
        # All boxes, belongs to unique image
        boxes = torch.as_tensor(all_rows[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32).to(self.device)
        # Rescale box coordinates according to new size of image after transforms
        if self.input_size:
            for i in range(len(boxes)):
                boxes[i] = v2.functional.resize_bounding_boxes(
                    boxes[i], 
                    (orig_img_shape_y, orig_img_shape_x),
                    self.input_size
                )[0]
        # All labels, belongs to every box
        labels = torch.tensor(all_rows['label'].values).to(self.device)
        # Target dict with boxes and labels, belongs to one unique image
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        return img, target