import base64
from aip import AipFace
import os
import time

APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

face_path = '../img_align_celeba_croped'
imageType = "BASE64"
groupId = "AOA"
num = 0
for face in os.listdir(face_path):
    reg_face_path = os.path.join(face_path, face)
    image = str(base64.b64encode(open(reg_face_path, 'rb').read()), 'utf-8')
    userId = face[:-4]
    print(userId)
    """ 调用人脸注册 """
    result = client.addUser(image, imageType, groupId, userId)
    if result['error_msg'] == 'SUCCESS':
        num = num + 1
    # 如果API返回结果为18，表示当前查询操作过于频繁，等待2秒继续查询
    elif result['error_code'] == 18:
        print("Open api qps request limit reached, please wait...")
        time.sleep(2)
        result = client.addUser(image, imageType, groupId, userId)
        if result['error_msg'] == 'SUCCESS':
            num = num + 1
    # 其他返回结果
    else:
        print(result)

print('Register {} faces'.format(num))


