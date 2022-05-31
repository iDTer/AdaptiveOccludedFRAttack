import base64
import json
import os
import time

from aip import AipFace

APP_ID = '25986474'
API_KEY = '7aoVxPZ138Qn1hsN3wa7ONgG'
SECRET_KEY = 'b5a1o4HwURBrbZqDTI8UGvkg0wB6UieH'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)
imageType = "BASE64"
groupIdList = "testgroup"

""" 如果有可选参数 """
options = {}
options["match_threshold"] = 30
# options["quality_control"] = "NORMAL"
# options["liveness_control"] = "LOW"
# options["user_id"] = "233451"
options["max_user_num"] = 5
rec_face_path = '../images_300_iteration'
likelihood_json_path = '../likelihood.json'
write_txt_path = './result/images_300_result.txt'
if os.path.exists(write_txt_path):
    os.remove(write_txt_path)
file_handle = open(write_txt_path, 'a')


# Output:   likelihood_list     <class 'list'>
def read_likelihood_json():
    with open(likelihood_json_path) as f:
        likelihood_list = json.load(f)
        return likelihood_list


def print_result_table(client_result):
    top_5_face_str = client_result['result']['user_list']
    print("{}\t\t{}\t\t{}\t\t".format(
        "top",
        "user_id",
        "score"
    ))
    file_handle.write('top\t\tuser_id\t\tscore\t\t\n')
    for i in range(5):
        print("{}\t\t{}\t\t{}\t\t".format(
            i + 1,
            top_5_face_str[i]['user_id'],
            top_5_face_str[i]['score']
        ))
        file_handle.write(str(i + 1) + '\t\t' +
                          top_5_face_str[i]['user_id'] + '\t\t'
                          + str(top_5_face_str[i]['score']) + '\t\t\n')


# Input:   client_result, my_target     <class 'str'>
# Output:   count     <class 'int'>
def count_success_num(client_result, my_target):
    count = 0
    top_5_face_str = client_result['result']['user_list']
    for i in range(5):
        rec_usrId = top_5_face_str[i]['user_id']
        if rec_usrId == my_target:
            count = 1
            continue
    return count


if __name__ == '__main__':
    success_num = 0
    likelihood_list_from_json = read_likelihood_json()

    for face in os.listdir(rec_face_path):
        userId_gt = face[:-4]
        target_userId = likelihood_list_from_json[face][0][:-4]

        # 查找json中的目标
        print("{} ==> {}".format(face, target_userId + '.jpg'))
        file_handle.write(face + " ==> " + target_userId + '.jpg\n')
        face_path = os.path.join(rec_face_path, face)
        image = str(base64.b64encode(open(face_path, 'rb').read()), 'utf-8')

        """ 带参数调用人脸搜索 """
        result = client.search(image, imageType, groupIdList, options)
        # 如果API返回结果为识别成功
        if result['error_msg'] == 'SUCCESS':
            print_result_table(result)

            """判断攻击是否成功"""
            if count_success_num(result, target_userId):
                print("Congratulation, face rec has been hacked!")
                success_num = success_num + 1
        # 如果API返回结果为18，表示当前查询操作过于频繁，等待2秒继续查询
        elif result['error_code'] == 18:
            print("Open api qps request limit reached, please wait...")
            time.sleep(2)
            result = client.search(image, imageType, groupIdList, options)
            print_result_table(result)
        # 其他返回结果
        else:
            print(result)
    print('success_num is ' + str(success_num))
    file_handle.write('success_num is ' + str(success_num) + '\n')

