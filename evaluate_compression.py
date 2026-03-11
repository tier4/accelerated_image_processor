import glob
import os
import signal
import subprocess
import time
from enum import Enum, auto
from pathlib import Path

import rclpy
import rosbag2_py
import yaml
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from rclpy.node import Node
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image
from tqdm import tqdm

script_dir = Path().resolve().absolute()
# ==========================================
# 設定パラメータ（環境に合わせて変更してください）
# ==========================================
# INPUT_BAG_PATH = str(script_dir / 'ssd_mount_point' / 'rosbag_20230329' /
#                      'RinkaiFukutoshin-to-Sachiura-1' / 'camera0' /
#                      'rosbag2_2023_03_29-01_17_43')
# INPUT_BAG_PATH = str(script_dir / 'ssd_mount_point' / 'rosbag_20230329' /
#                      'rosbag2_2023_03_29-10_17_44-1_camera6')
INPUT_BAG_PATH = str(
    script_dir / "ssd_mount_point" / "rosbag_20230329" / "rosbag2_2023_03_29-13_42_59_camera7"
)
# INPUT_BAG_PATH = str('/tmp/input_bag')

# OUTPUT_BAG_DIR = str(script_dir / 'ssd_mount_point' / 'rosbag_20230329' /
#                      'compressed' / 'rosbag2_2023_03_29-01_17_43_camera0'
#                      )
# OUTPUT_BAG_DIR = str(script_dir / 'ssd_mount_point' / 'rosbag_20230329' /
#                      'compressed' / 'rosbag2_2023_03_29-10_17_44-1_camera6'
#                      )
OUTPUT_BAG_DIR = str(
    script_dir
    / "ssd_mount_point"
    / "rosbag_20230329"
    / "compressed"
    / "rosbag2_2023_03_29-13_42_59_camera7"
)
# OUTPUT_BAG_DIR = str('/tmp/output_rosbags')

# YAML_DIR = 'config_yamls'                    # パラメータのYAMLファイルがあるディレクトリ
YAML_DIR = str(script_dir / "configs")

# INPUT_TOPIC = '/sensing/camera/camera0/image_rect_color'  # 圧縮ノードへ送る入力トピック名
# OUTPUT_TOPIC = '/sensing/camera/camera0/image_rect_color/compressed'  # 圧縮ノードから受け取る出力トピック名
# INPUT_TOPIC = '/sensing/camera/camera6/image_raw'  # 圧縮ノードへ送る入力トピック名
# OUTPUT_TOPIC = '/sensing/camera/camera6/image_raw/compressed'  # 圧縮ノードから受け取る出力トピック名
INPUT_TOPIC = "/sensing/camera/camera7/image_raw"  # 圧縮ノードへ送る入力トピック名
OUTPUT_TOPIC = (
    "/sensing/camera/camera7/image_raw/compressed"  # 圧縮ノードから受け取る出力トピック名
)

COMPRESSION_NODE_CMD = [
    "ros2",
    "run",
    "accelerated_image_processor_ros",
    "accelerated_image_processor_ros_imgproc_node",  # 圧縮ノードの起動コマンド
]

PUBLISH_INTERVAL = 0.1  # キュー溢れを防ぐための送信間隔（秒）。0.1なら約10Hz
WAIT_TIMEOUT = 5.0  # 全データ送信後、これだけ秒数データが来なければ終了とみなす
# ==========================================


class CompressionType(Enum):
    CompressedImage = auto()
    FFMPEGPacket = auto()


class BagProcessor(Node):

    def __init__(self, output_bag_path, compression_type):
        super().__init__("bag_processor_node")

        match compression_type:
            case CompressionType.CompressedImage:
                OutputMsgType = CompressedImage
            case CompressionType.FFMPEGPacket:
                OutputMsgType = FFMPEGPacket

        # PublisherとSubscriberの作成
        self.publisher = self.create_publisher(Image, INPUT_TOPIC, 10)
        self.subscription = self.create_subscription(
            OutputMsgType, OUTPUT_TOPIC, self.listener_callback, 10
        )

        # 出力用rosbagのセットアップ
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py._storage.StorageOptions(
            uri=output_bag_path, storage_id="sqlite3"
        )
        converter_options = rosbag2_py._storage.ConverterOptions("", "")
        self.writer.open(storage_options, converter_options)

        # 出力トピックの情報をrosbagに登録
        msg_package_name = OutputMsgType.__module__.split(".")[0]
        msg_type_name = f"{msg_package_name}/msg/{OutputMsgType.__name__}"
        # msg_type_name = OutputMsgType.__module__.replace(
        #     '.msg._', '/msg/') + OutputMsgType.__name__
        topic_info = rosbag2_py._storage.TopicMetadata(
            name=OUTPUT_TOPIC, type=msg_type_name, serialization_format="cdr"
        )
        self.writer.create_topic(topic_info)

        self.last_msg_time = time.time()
        self.received_count = 0

    def listener_callback(self, msg):
        # 圧縮データを受け取ったらrosbagに書き込む
        timestamp = self.get_clock().now().nanoseconds
        self.writer.write(OUTPUT_TOPIC, serialize_message(msg), timestamp)
        self.last_msg_time = time.time()
        self.received_count += 1


def get_compression_type(yaml_path):
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    compressor_type_str = params["/**"]["ros__parameters"]["compressor"]["type"]
    if compressor_type_str.lower() == "jpeg":
        return CompressionType.CompressedImage
    else:
        return CompressionType.FFMPEGPacket


def process_single_yaml(yaml_path, output_bag_path):
    rclpy.init()
    compression_type = get_compression_type(yaml_path)
    node = BagProcessor(output_bag_path, compression_type)

    # 1. 圧縮ノードをサブプロセスとして起動（YAMLパラメータを指定）
    cmd = COMPRESSION_NODE_CMD + [
        "--ros-args",
        "--params-file",
        yaml_path,
        "-r",
        f"image_raw:={INPUT_TOPIC}",
        "-r",
        f"image_raw/compressed:={OUTPUT_TOPIC}",
    ]

    node.get_logger().info(f"ノードを起動します: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, preexec_fn=os.setsid)

    try:
        # 2. 通信が確立するまで待機（QoS設定の完了待ち）
        node.get_logger().info("通信の確立を待機しています...")
        while rclpy.ok():
            pub_count = node.count_subscribers(INPUT_TOPIC)
            sub_count = node.count_publishers(OUTPUT_TOPIC)
            if pub_count > 0 and sub_count > 0:
                node.get_logger().info("通信が確立しました！")
                time.sleep(1.0)  # 念のため追加で少し待つ
                break
            rclpy.spin_once(node, timeout_sec=0.1)

        # 3. 入力rosbagからデータを読み込み、ゆっくりPublish
        node.get_logger().info("入力データの送信を開始します...")
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py._storage.StorageOptions(
            uri=INPUT_BAG_PATH, storage_id="sqlite3"
        )
        converter_options = rosbag2_py._storage.ConverterOptions("", "")
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        progress_bar = tqdm(total=len(topic_types))
        sent_count = 0
        while reader.has_next():
            progress_bar.update(1)
            topic, data, t = reader.read_next()
            if topic == INPUT_TOPIC:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)

                node.publisher.publish(msg)
                sent_count += 1

                # ROSのイベントを処理しつつ、指定した間隔だけ待つ（キュー溢れ防止）
                rclpy.spin_once(node, timeout_sec=PUBLISH_INTERVAL)
                # start_time = time.time()
                # while time.time() - start_time < PUBLISH_INTERVAL:
                #     rclpy.spin_once(node, timeout_sec=0.01)

        node.get_logger().info(
            f"全ての入力データ({sent_count}件)の送信が完了しました。出力の完了を待ちます..."
        )
        progress_bar.close()

        # 4. 終了待機（タイムアウト方式）
        node.last_msg_time = time.time()  # タイムアウトのタイマーをリセット
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = time.time() - node.last_msg_time
            if elapsed > WAIT_TIMEOUT:
                node.get_logger().info(
                    f"{WAIT_TIMEOUT}秒間データを受信しなかったため、処理を終了します。"
                )
                break

        node.get_logger().info(f"保存された圧縮データ: {node.received_count}件")

    finally:
        # 5. クリーンアップとノードの終了
        # process.terminate()
        # process.wait()  # 完全に終了するまで待つ
        if process.poll() is None:  # Process is still working
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait()  # 完全に終了するまで待つ

        node.writer = None  # rosbagを安全に閉じる
        node.destroy_node()
        rclpy.shutdown()
        time.sleep(1.0)  # 次のノード起動前に少し余裕を持たせる（ポートの解放待ちなど）


def main():
    if not os.path.exists(OUTPUT_BAG_DIR):
        os.makedirs(OUTPUT_BAG_DIR)

    yaml_files = glob.glob(os.path.join(YAML_DIR, "*.yaml"))
    if not yaml_files:
        print(f"{YAML_DIR}: YAMLファイルが見つかりません。パスを確認してください。")
        return

    for yaml_path in yaml_files:
        filename = os.path.basename(yaml_path)
        bag_name = f"result_{os.path.splitext(filename)[0]}"
        output_bag_path = os.path.join(OUTPUT_BAG_DIR, bag_name)

        print(f"--- 処理開始: {yaml_path} ---")
        process_single_yaml(yaml_path, output_bag_path)
        print(f"--- 処理完了: 結果は {output_bag_path} に保存されました ---\n")


if __name__ == "__main__":
    main()
