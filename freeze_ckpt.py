import tensorflow as tf
import pdb
# 创建推理图

# 导入ckpt变量
saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, tf.train.latest_checkpoint('./ckpt_gpu'))

# ckpt推理结果
pre = sess.run([], feed_dict={})
print(pre)

# ckpt转pb
# # 修改节点名称
# graph_def = tf.get_default_graph().as_graph_def()
# for node in graph_def.node:
# 	print(node.op)
# 	if node.op == 'RefSwitch':
# 		node.op = 'Switch'
# 		for  index in range(len(node.input)):
# 			if 'moving_' in node.input[index]:
# 				node.input[index] = node.input[index] + '/read'
# 	elif node.op == 'AssighSub':
# 		node.op = 'Sub'
# 		if 'use_locking' in node.attr: del node.attr['use_locking']
# 	elif node.op == 'AssighAdd':
# 		node.op = 'Add'
# 		if 'use_locking' in node.attr: del node.attr['use_locking']
# 	elif  node.op == 'Assigh':
# 		node.op = 'Identity'
# 		if 'use_locking' in node.attr: del node.attr['use_locking']
# 		if 'validate_shape' in node.attr: del node.attr['validate_shape']
# 		if len(node.input) == 2:
# 			node.input[0] = node.input[1]
# 			del node.input[1]
# 			# 修改end
output_node_names = ''
converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, 
	input_graph_def = sess.graph.as_graph_def(), 
	output_node_names = output_node_names.split(','))

with tf.gfile.GFile('model.pb', 'wb') as f:
	f.write(converted_graph_def.SerializeToString())

sess.close()
# 加载pb推理
with tf.Graph().as_default():
	output_graph_def = tf.compat.v1.GraphDef()
	# open pb
	with open('model.pb', 'rb') as f:
		output_graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(output_graph_def, name = "")

	with tf.Session() as sess:
		init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
		sess.run(init)
		input_1 = sess.graph.get_tensor_by_name('Placeholder:0')
		out = sess.graph.get_tensor_by_name('')
		pre = sess.run(out, feed_dict={input_1:''})
		print(pre)