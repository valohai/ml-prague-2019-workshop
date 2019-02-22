import tensorflow as tf

# Our warm-up example turned into single-process "cluster".
server = tf.train.Server.create_local_server()

greeting = tf.constant('Hello, distributed Prague!')
with tf.Session(server.target) as sess:
    result = sess.run(greeting)
    print(result)

# We can also use the same cluster configuration with other sessions.
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
with tf.Session(server.target) as sess:
    first_result = sess.run(add, feed_dict={a: 1, b: 2})
    print(f'1 + 2 = {first_result}')
    second_result = sess.run(add, feed_dict={a: 3, b: 4})
    print(f'3 + 4 = {second_result}')
