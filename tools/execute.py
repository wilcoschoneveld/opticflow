import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('models/checkpoint.ckpt.meta')
    saver.restore(sess, 'models/checkpoint.ckpt')

    print(sess.run('prediction:0', {'input:0': [[1, 0]]}))
