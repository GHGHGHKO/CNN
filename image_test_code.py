# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
import glob

#imagePath = './test_data/frame         0.jpg'                                      # 추론을 진행할 이미지 경로
#imagePath = './test_data/frame         0.jpg'                                      # 추론을 진행할 이미지 경로
#imagePath = './test_data/' + argv[1]                                     # 추론을 진행할 이미지 경로
#modelFullPath = '/tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
#labelsFullPath = '/tmp/output_labels.txt'                                   # 읽어들일 labels 파일 경로

an = 0
nyeong = 0
ha = 0
se = 0
yo = 0
cnt = 0

def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    global an
    global nyeong
    global ha
    global se
    global yo
    global cnt
    
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id] 
            print('%s (score = %.5f)' % (human_string, score))
            if str(r"b'an\n'") == human_string:
                an += score
            elif str(r"b'nyeong\n'") == human_string:
                nyeong += score
            elif str(r"b'ha\n'") == human_string:
                ha += score
            elif str(r"b'se\n'") == human_string:
                se += score
            else:
                yo += score
            
        #print (labels)
            
        #print (human_string)

        answer = labels[top_k[0]]
        cnt += 1
        if cnt == len(glob.glob('./test_data/*.jpg')):
            unsort = { 'an':an / cnt, 
                      'nyeong':nyeong / cnt / cnt, 
                      'ha':ha / cnt, 
                      'se':se / cnt, 
                      'yo':yo / cnt 
                     }
            print ('*' * 20)
            print (sorted(unsort.items(), key = lambda item: item[1])[-1])
            print ('*' * 20)
        return answer


if __name__ == '__main__':
    
     #run_inference_on_image()
    
    path = './test_data'

    file_list = os.listdir(path)

    file_list.sort()
    
    for i in file_list:
        if i.find('jpg') is not -1:
            print (i)
            imagePath = './test_data/' + i                                      # 추론을 진행할 이미지 경로
            modelFullPath = '/tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
            labelsFullPath = '/tmp/output_labels.txt'                                   # 읽어들일 labels 파일 경로

            print ('*' * 10 + i + '*' * 10)
            
            run_inference_on_image()
