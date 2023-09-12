import json, cv2, os, shutil
import numpy as np
import pdb

# Define Categories
cats = ['Car', 'Van', 'Truck', 'Tram', 'Pedestrian', 'Cyclist', 'Person_sitting', 'Misc', 'DontCare']
cat2id = {cat: i for i, cat in enumerate(cats)}
cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i})




data_path = '/home/iis/workspace/dataset/kitti3d'
image_dir = os.path.join(data_path, 'training', 'image_2')



num_samples = 200

#KITTI_P2_ProjMatix = np.zeros((3,4)).astype(np.float32) # <----- IMPORTANT

# unrectified
cameraMatrix = np.array([
    [996.60897068, 0, 0],
    [0, 998.66191895, 545.70492376],
    [0, 0, 1]
]).astype(np.float64)


img_rectify = True
'''
newCameraMatrix = np.array([
    [6.23293274e+02, 0, 1.19149990e+03],
    [0, 4.19680023e+02, 6.23797731e+02],
    [0, 0, 1]
]).astype(np.float64)
'''
newCameraMatrix = np.array([
    [2.15767456e+02, 0.00000000e+00, 5.45930045e-02],
    [0.00000000e+00, 1.31238235e+02, 8.88857263e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
]).astype(np.float64)
dist = np.array([[-0.40264286, 0.20029007, 0.00079995, 0.00055251, -0.05440088]]).astype(np.float64)

RT = np.array([[0,0,0]]).T
KITTI_P2_ProjMatix = np.hstack((newCameraMatrix, RT))




home_path = '.'
subset = 'val'
split = 'val1'
image_dir_des = os.path.join(home_path, split, subset, 'image_2')
os.makedirs(image_dir_des, exist_ok=True)

ret = {'images': [], 'annotations': [], 'categories': cat_info}



vid_capture = cv2.VideoCapture('/home/iis/workspace/bts/pytorch/20230822_230014-00.00.22.922-00.01.09.469.mp4')
if not vid_capture.isOpened():
    print('error opening the *.mp4')



newmtx_list = []
roi_list = []


image_id = -1
#with open(os.path.join(home_path, split, subset+'.txt'), 'r') as image_set:
while vid_capture.isOpened():
    gotframe, frame = vid_capture.read()
    if not gotframe:
        break
    if image_id == num_samples:
        break
    
    if img_rectify:
        h, w = frame.shape[:2]
        #print(cameraMatrix)
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix,
            dist,
            (w, h),
            1,
            (w, h)
        )
        #print("...")
        frame = cv2.undistort(
            frame,
            cameraMatrix,
            dist,
            None,
            newCameraMatrix
        )
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        print('roi:')
        print(roi)
        print('newCameraMatrix:')
        print(newCameraMatrix)

        newmtx_list.append(newCameraMatrix)
        roi_list.append(roi)

    

    
    image_id += 1
        

    # save frame
    file_name = '%06d.png'%(image_id)
    image_path_des = os.path.join(image_dir_des, file_name)
    cv2.imwrite(image_path_des, frame)

    # write json
    image = frame
    image_height, image_width = image.shape[:2]
    calib = KITTI_P2_ProjMatix
    image_info = {
        'file_name' : file_name,
        'id' : image_id,
        'calib' : calib.tolist(),
        'height' : image_height,
        'width' : image_width
    }
    #print(image_info)
    ret['images'].append(image_info)

print('### image num: ', len(ret['images']))
json_path = os.path.join(home_path, split, 'KITTI_'+split+'_'+subset+'.json')
with open(json_path, 'w') as f:
    json.dump(ret, f)


vid_capture.release()


pdb.set_trace()

print("FINISH")
