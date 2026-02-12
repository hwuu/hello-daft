"""
Lance 存储操作模块。

封装数据湖中数据集和模型的增删查操作。
数据湖目录结构：
    .ai_platform/
    ├── datasets/    # 数据集（如 mnist_clean.lance）
    └── models/      # 模型（如 mnist_cnn_v1.lance）
"""

import logging
import shutil
from pathlib import Path

import daft

logger = logging.getLogger(__name__)


class Storage:
    """Lance 数据湖存储管理器。

    负责扫描 .ai_platform/ 目录下的 .lance 文件，
    提供列出、查看详情、删除等操作。
    """

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.datasets_path = self.storage_path / "datasets"
        self.models_path = self.storage_path / "models"
        # 确保目录存在
        self.datasets_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"存储管理器初始化完成，数据湖路径: {self.storage_path}")

    def _list_tables(self, directory: Path) -> list[dict]:
        """扫描目录下所有 .lance 文件，返回元信息列表。"""
        results = []
        for p in sorted(directory.glob("*.lance")):
            try:
                df = daft.read_lance(str(p))
                schema = {f.name: str(f.dtype) for f in df.schema()}
                num_rows = df.count_rows()
                results.append({
                    "id": p.stem,
                    "path": str(p),
                    "schema": schema,
                    "num_rows": num_rows,
                })
            except Exception as e:
                # Lance 文件损坏或格式不兼容时，仍然列出但标记错误
                logger.warning(f"无法读取 {p}: {e}")
                results.append({"id": p.stem, "path": str(p), "error": "unreadable"})
        logger.info(f"扫描 {directory.name}/，找到 {len(results)} 个表")
        return results

    def _get_table(self, directory: Path, table_id: str) -> dict | None:
        """获取单个 Lance 表的详情（schema + 行数）。"""
        p = directory / f"{table_id}.lance"
        if not p.exists():
            logger.warning(f"表不存在: {p}")
            return None
        df = daft.read_lance(str(p))
        schema = {f.name: str(f.dtype) for f in df.schema()}
        num_rows = df.count_rows()
        logger.info(f"读取表 {table_id}: {num_rows} 行, 字段 {list(schema.keys())}")
        return {
            "id": table_id,
            "path": str(p),
            "schema": schema,
            "num_rows": num_rows,
        }

    def _delete_table(self, directory: Path, table_id: str) -> bool:
        """删除 Lance 表（整个 .lance 目录）。"""
        p = directory / f"{table_id}.lance"
        if not p.exists():
            logger.warning(f"删除失败，表不存在: {p}")
            return False
        shutil.rmtree(p)
        logger.info(f"已删除表: {p}")
        return True

    # --- 数据集操作 ---

    def list_datasets(self) -> list[dict]:
        """列出数据湖中所有数据集。"""
        return self._list_tables(self.datasets_path)

    def get_dataset(self, dataset_id: str) -> dict | None:
        """获取数据集详情。"""
        return self._get_table(self.datasets_path, dataset_id)

    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集。"""
        return self._delete_table(self.datasets_path, dataset_id)

    # --- 模型操作 ---

    def list_models(self) -> list[dict]:
        """列出数据湖中所有模型。"""
        return self._list_tables(self.models_path)

    def get_model(self, model_id: str) -> dict | None:
        """获取模型详情。"""
        return self._get_table(self.models_path, model_id)

    def delete_model(self, model_id: str) -> bool:
        """删除模型。"""
        return self._delete_table(self.models_path, model_id)
