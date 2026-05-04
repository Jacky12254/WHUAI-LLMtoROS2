
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

test_ratio = 0.1  # 测试集比例
val_ratio = 0.1   # 验证集比例
imgpath = r'C:\Users\Agora\robohome\data\images'  # 图片路径
txtpath = r'C:\Users\Agora\robohome\data\labels'      # 标签路径

BASE_DIR = Path(r'C:\Users\Agora\robohome\data\dataset')  # 输出目录

SUPPORTED_IMG_EXTS = ['.jpg', '.jpeg', '.png']


folders = {
    'train': {'img': BASE_DIR / 'images' / 'train', 'lbl': BASE_DIR / 'labels' / 'train'},
    'val':   {'img': BASE_DIR / 'images' / 'val',   'lbl': BASE_DIR / 'labels' / 'val'},
    'test':  {'img': BASE_DIR / 'images' / 'test',  'lbl': BASE_DIR / 'labels' / 'test'},
}


for f in folders.values():
    f['img'].mkdir(parents=True, exist_ok=True)
    f['lbl'].mkdir(parents=True, exist_ok=True)


txt_files = [f for f in os.listdir(txtpath) if f.endswith('.txt')]



def find_image_file(stem, img_dir, exts):
    for ext in exts:
        img_path = os.path.join(img_dir, stem + ext)
        if os.path.exists(img_path):
            return img_path, ext
    return None, None


paired_files = []
missing_imgs = []

for txt_file in txt_files:
    stem = txt_file[:-4]
    img_file, _ = find_image_file(stem, imgpath, SUPPORTED_IMG_EXTS)
    if img_file:
        paired_files.append((txt_file, img_file))
    else:
        missing_imgs.append(txt_file)

if missing_imgs:
    print(f"{len(missing_imgs)} 个标签文件找不到对应图像，例如：{missing_imgs[:3]}")



txt_list = [pair[0] for pair in paired_files]
trainval_txt, test_txt = train_test_split(txt_list, test_size=test_ratio, random_state=42)
relative_val_ratio = val_ratio / (1 - test_ratio)
train_txt, val_txt = train_test_split(trainval_txt, test_size=relative_val_ratio, random_state=42)

split_map = {}
for t in train_txt:
    split_map[t] = 'train'
for v in val_txt:
    split_map[v] = 'val'
for te in test_txt:
    split_map[te] = 'test'

def copy_file_pair(txt_file, img_file, split):
    # 目标路径
    dst_img = folders[split]['img'] / os.path.basename(img_file)
    dst_lbl = folders[split]['lbl'] / txt_file

    shutil.copy(img_file, dst_img)
    shutil.copy(os.path.join(txtpath, txt_file), dst_lbl)

for txt_file, img_file in paired_files:
    split = split_map[txt_file]
    copy_file_pair(txt_file, img_file, split)

total_files = len(paired_files)
print(f"训练集: {len(train_txt)}")
print(f"验证集: {len(val_txt)} ")
print(f"测试集: {len(test_txt)} ")
print("数据集已保存到:", BASE_DIR)
