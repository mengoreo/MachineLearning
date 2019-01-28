import tensorflow as tf
x1 = tf.constant(5)
x2 = tf.constant(3)
result = tf.multiply(x1, x2)
# result = x1 * x2
print(result)

# sess = tf.Session()
# print(sess.run(result))
with tf.Session() as sess:
    out = sess.run(result)
    print(out)

# out is now a python variable
print(out)
# print(sess.run(result)) out of session
