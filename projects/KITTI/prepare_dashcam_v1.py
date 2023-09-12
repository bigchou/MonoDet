import json, cv2, os, shutil
import numpy as np




cats = ['Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Cyclist', 'Person_sitting', 'Misc', 'DontCare']
cat2id = {cat: i for i, cat in enumerate(cats)}
cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i})




data_path = '/home/iis/workspace/dataset/kitti3d'
split = 'val1'
image_dir = os.path.join(data_path, 'training', 'image_2')




#KITTI_P2_ProjMatix = np.zeros((3,4)).astype(np.float32) # <----- IMPORTANT
KITTI_P2_ProjMatix = np.array([
    [721.5377197265625, 0, 609.559326171875, 44.85728073120117],
    [0, 721.5377197265625, 172.85400390625, 0.2163791060447693],
    [0, 0, 1, 0.0027458840049803257]
]).astype(np.float32)




home_path = '.'
subset = 'val'
split = 'val1'
image_dir_des = os.path.join(home_path, split, subset, 'image_2')
os.makedirs(image_dir_des, exist_ok=True)

ret = {'images': [], 'annotations': [], 'categories': cat_info}
image_id = -1
with open(os.path.join(home_path, split, subset+'.txt'), 'r') as image_set:
    for line in image_set:
        line = line.strip()
        image_id += 1
        

        # copy image
        image_path = os.path.join(image_dir, line+'.png')
        file_name = '%06d.png'%(image_id)
        image_path_des = os.path.join(image_dir_des, file_name)
        shutil.copy(image_path, image_path_des)

        # write json
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        calib = KITTI_P2_ProjMatix
        image_info = {
            'file_name' : file_name,
            'id' : image_id,
            'calib' : calib.tolist(),
            'height' : image_height,
            'width' : image_width
        }
        ret['images'].append(image_info)

print('### image num: ', len(ret['images']))
json_path = os.path.join(home_path, split, 'KITTI_'+split+'_'+subset+'.json')
with open(json_path, 'w') as f:
    json.dump(ret, f)

