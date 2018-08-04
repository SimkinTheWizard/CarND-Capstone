from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os, rospkg
import rospy
import cv2

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        self.is_site = is_site

        # I borrowed some of the code here from the TF object detection API
        config = 'sim'
        if is_site:
            config = 'real'   
        rospack = rospkg.RosPack()
        PATH_TO_CKPT = os.path.join(rospack.get_path("tl_detector"), "light_classification/trained", config ,"frozen/frozen_inference_graph.pb") 
        #PATH_TO_CKPT = os.path.join(rospack.get_path("tl_detector"), "light_classification/trained", config ,"frozen/optimized_graph.pb") 
        PATH_TO_LABELS = os.path.join(rospack.get_path("tl_detector"), "light_classification/trained/object-detection.pbtxt") 

        NUM_CLASSES = 4
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        #categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        #category_index = label_map_util.create_category_index(categories)
        self.sess = tf.Session(graph=self.detection_graph)
        pass

    # helper function from - TF object detection API
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
            """Determines the color of the traffic light in the image
    
            Args:
                image (cv::Mat): image containing the traffic light

            Returns:
                int: ID of traffic light color (specified in styx_msgs/TrafficLight)

            """
      
            #TODO implement light color prediction
            #cv2.imshow('light', image)
            #cv2.waitKey(1)
            #with self.detection_graph.as_default():
            #with self.sess as sess:
            sess = self.sess
            if True:
                ops = self.detection_graph.get_operations()
                #ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}

                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        #tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                        tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                detected_class = output_dict['detection_classes'][0]
                detection_acc  = output_dict['detection_scores'][0]
                #rospy.loginfo('detection accuracy: '+ str(detection_acc))
                if detection_acc < 0.95:
                    return TrafficLight.UNKNOWN
                else:
                    if detected_class == 1:
                        rospy.loginfo('detection GREEN: '+ str(detection_acc))
                        return TrafficLight.GREEN
                    if detected_class == 2:
                        rospy.loginfo('detection RED: '+ str(detection_acc))
                        return TrafficLight.RED
                    if detected_class == 3:
                        rospy.loginfo('detection YELLOW: '+ str(detection_acc))
                        return TrafficLight.YELLOW
                    if detected_class == 4:
                        rospy.loginfo('detection UNKNOWN: '+ str(detection_acc))
                        return TrafficLight.UNKNOWN
            return TrafficLight.UNKNOWN
		
