import tensorflow as tf

samples_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
samples_y = [[1, 0], [0, 1], [0, 1], [1, 0]]

x = tf.placeholder(tf.float32, shape=[None, 2], name='input')
y_ = tf.placeholder(tf.float32, shape=[None, 2], name='correct')

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random_normal(shape=[2, 3]), name="weights")
    b1 = tf.Variable(tf.random_normal(shape=[3]), name="bias")
    h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random_normal(shape=[3, 2]), name="weights")
    b2 = tf.Variable(tf.random_normal(shape=[2]), name="bias")
    y = tf.matmul(h1, W2) + b2

with tf.name_scope("cost"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

with tf.name_scope("train"):
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_op)

prediction = tf.argmax(tf.nn.softmax(y), 1)
true = tf.argmax(y_, 1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, true), tf.float32))

writer = tf.summary.FileWriter("xor_log", graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        loss, _ = sess.run([loss_op, train_op], feed_dict={x: samples_x, y_: samples_y})
        print(loss)

    acc = sess.run(accuracy, feed_dict={x: samples_x, y_: samples_y})
    print(acc)
