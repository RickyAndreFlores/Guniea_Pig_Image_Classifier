import tensorflow as tf
import random



def model(input, labels, alpha):


	# Convolutional layer 1
	a1 = tf.contrib.layers.conv2d(
		input,
		num_outputs=15,
		kernel_size=3,
		stride=2,
		padding='SAME',
		data_format=None,
		rate=1,
		activation_fn=tf.nn.leaky_relu
	)

	# View some of the outputs in tensorboard

	a1_slice = tf.slice(a1, [0, 0, 0, 5], [500,128,128,1] )
	tf.summary.image("a1", a1_slice, max_outputs=10)

	a1_slice2 = tf.slice(a1, [0, 0, 0, 6], [500, 128, 128, 1])
	tf.summary.image("a1", a1_slice2, max_outputs=10)

	a1_slice3 = tf.slice(a1, [0, 0, 0, 7], [500, 128, 128, 1])
	tf.summary.image("a1", a1_slice3, max_outputs=10)

	a1_slice4 = tf.slice(a1, [0, 0, 0, 8], [500, 128, 128, 1])
	tf.summary.image("a1", a1_slice4, max_outputs=10)

	a1_slice5 = tf.slice(a1, [0, 0, 0, 9], [500, 128, 128, 1])
	tf.summary.image("a1", a1_slice5, max_outputs=10)


	# a2 = tf.nn.max_pool
	a1_max = tf.nn.max_pool(a1,
	                        ksize=[1, 2, 2, 1],
	                        strides=[1, 1, 1, 1],
	                        padding='VALID',
	                        data_format='NHWC',
	                        name=None
	                        )

	# Normalizes activations
	a1_norm = tf.nn.local_response_normalization(
		a1_max,
		depth_radius=5,
		bias=1,
		alpha=1,
		beta=0.5,
		name=None
	)

	a1_norm_slice = tf.slice(a1, [0, 0, 0, 7], [500,127,127,1] )
	tf.summary.image("a1_max_norm", a1_norm_slice, max_outputs=10)

	a2 = tf.contrib.layers.conv2d(
		a1_norm,
		num_outputs=10,
		kernel_size=3,
		stride=1,
		padding='VALID',
		data_format=None,
		rate=1,
		activation_fn=tf.nn.leaky_relu
	)


	a2_norm = tf.nn.local_response_normalization(
		a2,
		depth_radius=5,
		bias=1,
		alpha=1,
		beta=0.5,
		name=None
	)

	a2_max = tf.nn.max_pool(a2_norm,
	                        ksize=[1, 2, 2, 1],
	                        strides=[1, 1, 1, 1],
	                        padding='VALID',
	                        data_format='NHWC',
	                        name=None
	                        )

	a2_norm_slice = tf.slice(a1, [0, 0, 0, 1], [500, 124, 124, 1])
	tf.summary.image("a2_max_norm", a2_norm_slice, max_outputs=10)

	a2_norm_slice2 = tf.slice(a1, [0, 0, 0, 2], [500, 124, 124, 1])
	tf.summary.image("a2_max_norm", a2_norm_slice2, max_outputs=10)

	a2_norm_slice3 = tf.slice(a1, [0, 0, 0, 3], [500, 124, 124, 1])
	tf.summary.image("a2_max_norm", a2_norm_slice3, max_outputs=10)

	a2_norm_slice4 = tf.slice(a1, [0, 0, 0, 5], [500, 124, 124, 1])
	tf.summary.image("a2_max_norm", a2_norm_slice4, max_outputs=10)

	a2_norm_slice5 = tf.slice(a1, [0, 0, 0, 6], [500, 124, 124, 1])
	tf.summary.image("a2_max_norm", a2_norm_slice5, max_outputs=10)

	a2_norm_slice6 = tf.slice(a1, [0, 0, 0, 8], [500, 124, 124, 1])
	tf.summary.image("a2_max_norm", a2_norm_slice6, max_outputs=10)


	# Fully connected layer 1
	a3 = tf.contrib.layers.fully_connected(
		tf.contrib.layers.flatten(a2_max),
		num_outputs=100,
		activation_fn=tf.nn.leaky_relu
	)


	# Fully connected layer 2
	a4 = tf.contrib.layers.fully_connected(
		a3,
		num_outputs=1,
		activation_fn=None)


	# Calculate accuracy

	a4_sigmoid = tf.nn.sigmoid(a4)

	predictions = (a4_sigmoid > .5)

	y_hat = tf.cast(predictions, "float")

	labels_expanded = tf.expand_dims(labels, 1)

	correct = tf.equal(y_hat, labels_expanded)
	accur = tf.reduce_mean(tf.cast(correct, "float"))

	# Create a summary of accuracy
	tf.summary.scalar("Accuracy", accur)

	# Computes the cross entropy for each example after putting output through a sigmoid function
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_expanded, logits=a4, name="loss")

	# Optimizer for regression/backward pass
	learn_ = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cross_entropy)

	return learn_, accur, predictions


def data():

	def read_labeled_image_list(image_list_file, path, filenames_list, labels_list):
		"""
		Produces a list of all filenames and their corresponding labels

		"""

		f = open(image_list_file, 'r')

		for line in f:
			label, filename, = line[:-1].split("___")
			filename = path + filename
			filenames_list.append(filename)
			labels_list.append(int(label))

		# A normal list of all filenames and their corresponding labels
		return filenames_list, labels_list

	def read_images_from_disk(input_queue):
		"""
		Returns:
		  Two tensors: the decoded image, and the string label.

		"""

		label = input_queue[1]

		file_contents = tf.read_file(input_queue[0])

		# Decode the image
		img_orig = tf.image.decode_image(file_contents, channels=3)

		# Resize the images, to keep them of uniform size and/or to lower the amount of data to process
		img_resized = tf.image.resize_image_with_pad(img_orig, 256, 256)
		img_resized.set_shape((256, 256, 3))

		return img_resized, label


	image_list = []
	label_list = []


	# Get guinea pig data
	path_correct = "../GunieaPigs/"
	image_list, label_list = read_labeled_image_list("guineapig_filenames.txt", path_correct, image_list, label_list)
	# Get chipmunk data
	path_incorrect = "../Chipmonks/"
	image_list, label_list = read_labeled_image_list("chipmonks_filenames.txt", path_incorrect, image_list, label_list)
	# get car data
	path_incorrect2 = "../Cars/"
	image_list, label_list = read_labeled_image_list("muscle_cars.txt", path_incorrect2, image_list, label_list)


	path_and_label = list(zip(image_list,label_list))
	random.shuffle(path_and_label)

	# 80% of data goes to train set
	eighty_floored = len(path_and_label) * 8 // 10

	test_set_lists = list(zip(*path_and_label[:eighty_floored]))
	cv_set_lists =list(zip(* path_and_label[eighty_floored:]))

	images_train = tf.convert_to_tensor(test_set_lists[0], dtype=tf.string)
	labels_train = tf.convert_to_tensor(test_set_lists[1], dtype=tf.float32)

	images_cv = tf.convert_to_tensor(cv_set_lists[0], dtype=tf.string)
	labels_cv = tf.convert_to_tensor(cv_set_lists[1], dtype=tf.float32)


	input_queue = tf.train.slice_input_producer([images_train, labels_train], shuffle=True, )

	image_train, label_train = read_images_from_disk(input_queue)


	batch_size = 500

	image_batch, label_batch = tf.train.shuffle_batch([image_train, label_train],
	                                                  batch_size=batch_size,
	                                                  capacity=10,
	                                                  min_after_dequeue=5,
	                                                  allow_smaller_final_batch=True)

	# create summary of batch to view in tensorflow
	tf.summary.image("train batch", image_batch, max_outputs=7)


	input_queue_cv = tf.train.slice_input_producer([images_cv, labels_cv], shuffle=True, )
	image_test, label_test = read_images_from_disk(input_queue_cv)




	image_batch_cv, label_batch_cv = tf.train.shuffle_batch([image_test, label_test],
	                                                  batch_size=batch_size,
	                                                  capacity=10,
	                                                  min_after_dequeue=5,
	                                                  allow_smaller_final_batch=True)

	tf.summary.image("image_batch CSV 2", image_batch_cv, max_outputs=7)




	return image_batch, label_batch, image_batch_cv, label_batch_cv



# Prepare data Tensor
image_batch, label_batch, image_cv, label_cv = data()

# Train model Tensor
learn_, accur, pred = model(image_batch, label_batch, alpha=.01)

# Run model on cross validation tensor
learn_cv, accur_cv, pred_cv = model(image_cv, label_cv, alpha = .01)

merged = tf.summary.merge_all()

init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

saver = tf.train.Saver()


# sess = tf.Session()
with tf.Session() as sess:

	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)


	saver.restore(sess, "PeanutProgress")


	epochs = 100

	for step in range(epochs):

		try:

			loss, accuracy, predictions = sess.run([learn_, accur, pred])

			print("\nEpoch {}".format(step + 1),
			      "\nAccuracy:", accuracy)

			# For debugging purposes, helps see what the model is predicting in a shorthand way
			positive_percentage = sum(predictions) / len(predictions)
			print("Percentage of prediction = 1", positive_percentage)


		except tf.errors.OutOfRangeError:
			print("End of dataset")  # "End of dataset"

			break

	save_path = saver.save(sess, "Saved_Peanut_Progress/PeanutProgress")


	try:
		loss, accuracy, predictions = sess.run([learn_cv, accur_cv, pred_cv])

		print("\nCross validation Accuracy:", accuracy)

	except tf.errors.OutOfRangeError:
		print("End of dataset")



	writer = tf.summary.FileWriter(f'logs/', sess.graph)

	summary_merged = sess.run(merged)

	writer.add_summary(summary_merged)

	coord.request_stop()
	coord.join(threads)
