import tensorflow as tf


x = tf.constant([[1.,1.,1.,1.],
				[2.,2.,2.,2.],
				[3.,3.,3.,3.],
				[4.,4.,4.,4.]])
x = tf.reshape(x, [1,4,4,1])

weight = tf.get_variable("weight", [3,3,1,1], initializer=tf.constant_initializer(1))

conv = tf.nn.conv2d(input=x, filter=weight, strides=[1,2,1,1], padding="SAME")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(x.eval())
	print(weight.eval())
	print(conv.eval())
	# print(same_pad.eval())