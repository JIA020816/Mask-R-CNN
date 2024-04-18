import os
import json

from lxml import etree
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from train_utils import convert_to_coco_api


class VOCInstances(Dataset):
    def __init__(self, voc_root, year="2012", txt_name: str = "train.txt", transforms=None):
        super().__init__()
        if isinstance(year, int):
            year = str(year)
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        if "VOCdevkit" in voc_root:
            root = os.path.join(voc_root, f"VOC{year}")
        else:
            root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        xml_dir = os.path.join(root, 'Annotations')
        mask_dir = os.path.join(root, 'SegmentationObject')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # read class_indict
        json_file = 'pascal_voc_indices.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            idx2classes = json.load(f)
            self.class_dict = dict([(v, k) for k, v in idx2classes.items()])

        self.images_path = []     # 存储图片路径
        self.xmls_path = []       # 存储xml文件路径
        self.xmls_info = []       # 存储解析的xml字典文件
        self.masks_path = []      # 存储SegmentationObject图片路径
        self.objects_bboxes = []  # 存储解析的目标boxes等信息
        self.masks = []           # 存储读取的SegmentationObject图片信息
        self.num_objs = []
        # 检查图片、xml文件以及mask是否都在
        images_path = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        xmls_path = [os.path.join(xml_dir, x + '.xml') for x in file_names]
        masks_path = [os.path.join(mask_dir, x + ".png") for x in file_names]
        for idx, (img_path, xml_path, mask_path) in enumerate(zip(images_path, xmls_path, masks_path)):
            assert os.path.exists(img_path), f"not find {img_path}"
            assert os.path.exists(xml_path), f"not find {xml_path}"
            assert os.path.exists(mask_path), f"not find {mask_path}"

            # 解析xml中bbox信息
            # with open(xml_path) as fid:
                # xml_str = fid.read()
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            obs_dict = parse_xml_to_dict(xml)["annotation"]  # 将xml文件解析成字典
            obs_bboxes = parse_objects(obs_dict, xml_path, self.class_dict, idx)  # 解析出目标信息
            num_objs = obs_bboxes["boxes"].shape[0]
            # print(num_objs,'44444')
            # print(obs_bboxes,"000000")


            # 读取SegmentationObject并检查是否和bboxes信息数量一致
            instances_mask = Image.open(mask_path)

            instances_mask = np.array(instances_mask)
            # np.set_printoptions(threshold=np.inf)  # 设置打印选项：输出数组元素数目上限为无穷
            # print(instances_mask.shape)
            #
            # instances_mask[instances_mask == 255] = 1  # 255为背景或者忽略掉的地方，这里为了方便直接设置为背景(0)
            # np.set_printoptions(threshold=np.inf)  # 设置打印选项：输出数组元素数目上限为无穷
            # print(instances_mask)
            #
            # # 需要检查一下标注的bbox个数是否和instances个数一致
            # num_instances = instances_mask.max()
            # print(num_objs,num_instances,"000000")
            #
            #
            # if num_objs != num_instances:
            #     print(f"warning: num_boxes:{num_objs} and num_instances:{num_instances} do not correspond. "
            #           f"skip image:{img_path}")
            #     continue

            self.images_path.append(img_path)
            self.xmls_path.append(xml_path)
            self.xmls_info.append(obs_dict)
            self.masks_path.append(mask_path)
            self.objects_bboxes.append(obs_bboxes)
            self.masks.append(instances_mask)
            self.num_objs.append(num_objs)

        self.transforms = transforms
        self.coco = convert_to_coco_api(self)

    def parse_mask(self, idx: int):
        # 获取指定索引的掩膜
        mask = self.masks[idx]

        # 找到掩膜中的最大值，即目标的数量
        c = self.num_objs[idx]  # 有几个目标最大索引就等于几
        # print(self.num_objs[idx],'3333')
        # 初始化掩膜列表
        masks = []

        # 对每个目标的mask单独使用一个channel存放
        for i in range(1, c + 1):
            # 将每个目标的掩膜添加到masks列表中
            masks.append((mask == 1).astype(np.uint8))##############

        # 如果没有目标，返回None
        if not masks:
            print(f"No mask found for index {idx}. Returning None.")
            return None

        # 如果有目标，堆叠掩膜数据
        masks = np.stack(masks, axis=0)

        # 将NumPy数组转换为PyTorch张量
        return torch.as_tensor(masks, dtype=torch.uint8)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images_path[idx]).convert('RGB')
        target = self.objects_bboxes[idx]
        masks = self.parse_mask(idx)
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_path)

    def get_height_and_width(self, idx):
        """方便统计所有图片的高宽比例信息"""
        # read xml
        data = self.xmls_info[idx]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def get_annotations(self, idx):
        """方便构建COCO()"""
        data = self.xmls_info[idx]
        h = int(data["size"]["height"])
        w = int(data["size"]["width"])
        target = self.objects_bboxes[idx]
        masks = self.parse_mask(idx)
        target["masks"] = masks
        return target, h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def parse_objects(data: dict, xml_path: str, class_dict: dict, idx: int):
    """
    解析出bboxes、labels、iscrowd以及areas等信息
    Args:
        data: 将xml解析成dict的Annotation数据
        xml_path: 对应xml的文件路径
        class_dict: 类别与索引对应关系
        idx: 图片对应的索引

    Returns:
        A dictionary containing boxes, labels, iscrowd, image_id, and area.
    """
    boxes = []
    labels = []
    iscrowd = []
    area = []


    for obj in data.get("object", []):
        xmin = float(obj["bndbox"]["xmin"])
        ymin = float(obj["bndbox"]["ymin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymax = float(obj["bndbox"]["ymax"])

        # 检查bbox的尺寸是否有效
        if xmax <= xmin or ymax <= ymin:
            print(f"Warning: In '{xml_path}', there are some bbox w/h <= 0. Skipping.")
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        try:
            # 尝试获取类别索引并添加到labels列表中
            labels.append(int(class_dict[obj["name"]]))
        except KeyError:
            print(f"KeyError: '{obj['name']}' is not a key in class_dict for file '{xml_path}'. Skipping.")
            continue
        except ValueError as e:
            print(f"ValueError: The value for '{obj['name']}' may not be convertible to an integer for file '{xml_path}'. Skipping.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e} for file '{xml_path}'. Skipping.")

        if "difficult" in obj:
            iscrowd.append(int(obj["difficult"]))
        else:
            iscrowd.append(0)

    # 将列表转换为torch.Tensor
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.tensor([], dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.tensor([], dtype=torch.int64)
    iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.tensor([], dtype=torch.int64)
    image_id_tensor = torch.tensor([idx], dtype=torch.int64)
    area_tensor = torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32) if boxes else torch.tensor([], dtype=torch.float32)

    # print(class_dict.keys())

    return {"boxes": boxes_tensor,
            "labels": labels_tensor,
            "iscrowd": iscrowd_tensor,
            "image_id": image_id_tensor,
            "area": area_tensor}

if __name__ == '__main__':
    dataset = VOCInstances(voc_root="./VOCdevkit")
    print(len(dataset))
    d1 = dataset[0]
    # print(d1)
