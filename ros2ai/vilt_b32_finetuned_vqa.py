# Copyright 2023 Taiga Takano
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers import ViltProcessor, ViltForQuestionAnswering

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv_bridge

class ros2aiNode(Node):
    def __init__(self):
        self.declare_parameter('cache_dir', '/workspace/cache')
        self.declare_parameter('model_name', 'dandelin/vilt-b32-finetuned-vqa')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('question', 'How many human are there?')

        cache_dir = self.get_parameter('cache_dir').value
        model_name = self.get_parameter('model_name').value
        self.device = self.get_parameter('device').value
        self.question = self.get_parameter('question').value

        self.processor = ViltProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)

        super().__init__('vilt_b32_finetuned_vqa')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,0)
        self.subscription
        self.bridge = cv_bridge.CvBridge()
        self.publisher = self.create_publisher(String, 'result', 10)

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        encoding = self.processor(cv_image, self.question, return_tensors="pt").to(self.device)

        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        ans = self.model.config.id2label[idx]
        self.get_logger().info(ans)
        self.publisher.publish(String(data=ans))

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = ros2aiNode()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
