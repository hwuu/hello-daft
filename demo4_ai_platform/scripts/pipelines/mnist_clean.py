"""
MNIST 数据清洗脚本。

下载 MNIST 数据集，归一化像素值，写入 Lance 格式。
实现 run(input_path, output_path, params) 接口，供 Executor 调用。

数据流: MNIST IDX 文件 -> 归一化 -> Arrow Table -> Lance 文件
输出 schema: image(List[Float64]), label(Int64), split(String)
"""

import logging
import struct
from pathlib import Path

import daft
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


def run(input_path: str, output_path: str, params: dict) -> dict:
    """清洗 MNIST 数据并写入 Lance。

    Args:
        input_path: MNIST IDX 文件目录，或 "download" 自动下载
        output_path: Lance 输出路径
        params: {"normalize": bool} — 是否归一化像素值到 [0, 1]
    Returns:
        统计信息: {"total_records": int, "train_records": int, "test_records": int}
    """
    input_dir = Path(input_path)

    # 如果指定 "download" 或目录不存在，自动下载 MNIST 数据
    if input_path == "download" or not input_dir.exists():
        logger.info(f"数据目录不存在，开始下载 MNIST 数据到: {input_dir}")
        input_dir = _download_mnist(input_dir)

    # 读取 IDX 格式的图像和标签文件
    logger.info(f"读取 MNIST IDX 文件: {input_dir}")
    train_images = _read_idx_images(input_dir / "train-images-idx3-ubyte")
    train_labels = _read_idx_labels(input_dir / "train-labels-idx1-ubyte")
    test_images = _read_idx_images(input_dir / "t10k-images-idx3-ubyte")
    test_labels = _read_idx_labels(input_dir / "t10k-labels-idx1-ubyte")

    # 将 28x28 图像展平为 784 维向量
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)

    # 归一化像素值: [0, 255] -> [0.0, 1.0]
    if params.get("normalize", True):
        logger.info("归一化像素值到 [0, 1]")
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

    # 合并训练集和测试集，用 split 列区分
    n_train = len(train_labels)
    n_test = len(test_labels)
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_splits = ["train"] * n_train + ["test"] * n_test

    # 构建 Arrow Table 并写入 Lance
    logger.info(f"构建 Arrow Table: {n_train + n_test} 行")
    table = pa.table({
        "image": [row.tolist() for row in all_images],  # 每行是 784 维浮点数组
        "label": all_labels.tolist(),                     # 0-9 的整数标签
        "split": all_splits,                              # "train" 或 "test"
    })

    df = daft.from_arrow(table)
    df.write_lance(output_path, mode="overwrite")
    logger.info(f"数据已写入 Lance: {output_path}")

    return {
        "total_records": n_train + n_test,
        "train_records": n_train,
        "test_records": n_test,
    }


def _read_idx_images(path: Path) -> np.ndarray:
    """读取 IDX 格式的图像文件。

    IDX 文件格式（大端序）:
    - 4 字节 magic number (2051)
    - 4 字节 图像数量
    - 4 字节 行数 (28)
    - 4 字节 列数 (28)
    - 后续为像素数据 (uint8)

    Returns:
        shape=(num, 28, 28) 的 uint8 数组
    """
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"无效的图像文件 magic number: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """读取 IDX 格式的标签文件。

    IDX 文件格式（大端序）:
    - 4 字节 magic number (2049)
    - 4 字节 标签数量
    - 后续为标签数据 (uint8, 0-9)

    Returns:
        shape=(num,) 的 uint8 数组
    """
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"无效的标签文件 magic number: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def _download_mnist(target_dir: Path) -> Path:
    """从 Google Cloud Storage 下载 MNIST 数据集。

    下载 4 个 .gz 文件并解压为 IDX 格式。
    如果文件已存在则跳过下载。
    """
    import gzip
    import urllib.request

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",   # 训练图像 (60000 张)
        "train-labels-idx1-ubyte.gz",   # 训练标签
        "t10k-images-idx3-ubyte.gz",    # 测试图像 (10000 张)
        "t10k-labels-idx1-ubyte.gz",    # 测试标签
    ]

    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in files:
        gz_path = target_dir / fname
        out_path = target_dir / fname.replace(".gz", "")
        if out_path.exists():
            logger.info(f"文件已存在，跳过: {out_path}")
            continue
        logger.info(f"下载: {fname}")
        urllib.request.urlretrieve(base_url + fname, gz_path)
        # 解压 .gz 文件
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        gz_path.unlink()  # 删除 .gz 文件

    logger.info(f"MNIST 数据下载完成: {target_dir}")
    return target_dir
