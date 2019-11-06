import tensorflow as tf
import numpy as np


X = tf.placeholder(tf.float32, shape=[None,4])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([4,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X,W) + b

model = tf.global_variables_initializer()
saver = tf.train.Saver()

avg_temp = float(input("mean temp: "))
min_temp = float(input("minimum temp: "))
max_temp = float(input("maximum temp: "))
rain_fall = float(input("rainfall: "))

with tf.Session() as sess:
    sess.run(model)
    save_path = './saved.cpkt'
    saver.restore(sess, save_path)

    data = ((avg_temp, min_temp, max_temp, rain_fall),)
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X:x_data})

    print(dict[0])




