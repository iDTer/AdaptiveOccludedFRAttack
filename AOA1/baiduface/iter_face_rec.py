import base64
import json
import os
import time

from aip import AipFace

APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
client = AipFace(APP_ID, API_KEY, SECRET_KEY)
imageType = "BASE64"
groupIdList = "AOA"

""" 如果有可选参数 """
options = {}
options["match_threshold"] = 50
options["max_user_num"] = 10

rec_face_path = '../iterarion_pic_images'
likelihood_json_path = '../likelihood.json'
write_txt_path = '../pic_res/images_res.txt'
if os.path.exists(write_txt_path):
    os.remove(write_txt_path)
file_handle = open(write_txt_path, 'a')


# Output:   likelihood_list     <class 'list'>
def read_likelihood_json():
    with open(likelihood_json_path) as f:
        likelihood_list = json.load(f)
        return likelihood_list


# def print_result_table(client_result, my_iter_num):
#     top_50_face_str = client_result['result']['user_list']
#     print('iteration_num', my_iter_num)
#     print("{}\t\t{}\t\t{}\t\t".format(
#         "top",
#         "user_id",
#         "score"
#     ))
#     file_handle.write('top\t\tuser_id\t\tscore\t\t\n')
#     for i in range(5):
#         print("{}\t\t{}\t\t{}\t\t".format(
#             i + 1,
#             top_50_face_str[i]['user_id'],
#             top_50_face_str[i]['score']
#         ))
#         file_handle.write(str(i + 1) + '\t\t' +
#                           top_50_face_str[i]['user_id'] + '\t\t'
#                           + str(top_50_face_str[i]['score']) + '\t\t\n')


def if_attack_success(client_result, my_target):
    top_face_str = client_result['result']['user_list']
    for i in range(50):
        rec_usrId = top_face_str[i]['user_id']
        if rec_usrId == my_target:
            return True
        else:
            return False


def handle_result(client_result, my_iter_num, my_target):
    top_face_str = client_result['result']['user_list']
    for i in range(10):
        rec_usrId = top_face_str[i]['user_id']
        if rec_usrId == my_target:
            print('iteration_num', my_iter_num)
            print("{}\t\t{}\t\t{}\t\t".format(
                "top",
                "user_id",
                "score"
            ))
            file_handle.write('iteration_num: ' + str(my_iter_num) + '\n')
            file_handle.write("top\t\tuser_id\t\tscore\t\t\n")
            print("{}\t\t{}\t\t{}\t\t".format(
                i + 1,
                top_face_str[i]['user_id'],
                top_face_str[i]['score']
            ))
            for j in range(50):
                file_handle.write(str(j+1) + "\t\t" + top_face_str[j]['user_id']
                                  + "\t\t" + str(top_face_str[j]['score']) + "\t\t\n")


if __name__ == '__main__':
    likelihood_list_from_json = read_likelihood_json()
    face_img = ''
    userId_get = face_img[:-4]

    target_userId = likelihood_list_from_json[face_img][0][:-4]
    print("{} ==> {}".format(face_img, target_userId + '.jpg'))
    iterable_list = os.listdir(rec_face_path)
    iterable_list.sort(key=lambda x: int(x.split('_')[0]))
    for face_name in iterable_list:
        face_path = os.path.join(rec_face_path, face_name)
        iter_num = face_name.split('_')[0]
        image = str(base64.b64encode(open(face_path, 'rb').read()), 'utf-8')
        """ 带参数调用人脸搜索 """
        result = client.search(image, imageType, groupIdList, options)
        # 如果API返回结果为识别成功
        if result['error_msg'] == 'SUCCESS':
            handle_result(result, iter_num, target_userId)

        # 如果API返回结果为18，表示当前查询操作过于频繁，等待2秒继续查询
        elif result['error_code'] == 18:
            print("Open api qps request limit reached, please wait...")
            time.sleep(2)
            result = client.search(image, imageType, groupIdList, options)
            handle_result(result, iter_num, target_userId)









