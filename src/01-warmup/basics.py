import tensorflow as tf

# At the low level, TensorFlow creates graphs of operations.
# Each operation is one node in the graph and previous operations
# must be completed before the operation can be started.

# After you have defined your operations, you start
# a session which opens communications between Python and
# the C++ runtime of TensorFlow.

# Run a graph with 1 constant operation.
greeting = tf.constant('Hello, Prague!')
with tf.Session() as sess:
    result = sess.run(greeting)
    print(result)

# Placeholder-operations are like variables, you define them on runtime.
# Add-operation depends on 2 operations, creating a small graph.
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
with tf.Session() as sess:
    first_result = sess.run(add, feed_dict={a: 1, b: 2})
    print('1 + 2 =', first_result)
    second_result = sess.run(add, feed_dict={a: 3, b: 4})
    print('3 + 4 =', second_result)

# It might feel strange that the operations weren't specifically
# assigned to a specific "graph", you just typed methods
# and it all seems to work.
# Reason for this is that TensorFlow will create a "default" graph for
# all unassigned operations.
g = tf.get_default_graph()
print(g.get_operations())  # => [1x const, 2x placeholder, 1x add] operations
# So even though these two groups of operations (subgraphs)
# have nothing connecting them, they belong into a same graph.

# Sometimes a single graph is suboptimal though.
# For example, training and inference frequently have separate graphs to save
# resources and make sure different operations are used for e.g. dropout.
graph_a = tf.Graph()
with graph_a.as_default():
    constant_in_a = tf.constant('constant in graph a')

graph_b = tf.Graph()
with graph_b.as_default():
    constant_in_b = tf.constant('constant in graph b')

with tf.Session(graph=graph_a) as sess:
    print(sess.run(constant_in_a))  # => 'constant in graph a'

with tf.Session(graph=graph_b) as sess:
    print(sess.run(constant_in_b))  # => 'constant in graph b'

print(g.get_operations())  # => default graph still only has the 4 originals
