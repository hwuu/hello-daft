"""
数据生成脚本 - 生成示例产品数据集

用法:
    python generate_data.py --size 100000 --output products.csv
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# 初始化 Faker
fake = Faker(['zh_CN'])
Faker.seed(42)
random.seed(42)

# 产品类别和子类别
CATEGORIES = {
    '电子产品': ['手机', '电脑', '平板', '耳机', '相机', '智能手表'],
    '家居用品': ['家具', '厨具', '床上用品', '装饰品', '收纳用品'],
    '服装': ['男装', '女装', '童装', '运动装', '鞋类'],
    '图书': ['小说', '技术书籍', '教材', '漫画', '杂志'],
    '食品': ['零食', '饮料', '生鲜', '调味品', '保健品'],
    '运动户外': ['健身器材', '户外装备', '运动服饰', '球类', '自行车'],
    '美妆': ['护肤品', '彩妆', '香水', '个人护理', '美容工具'],
    '玩具': ['益智玩具', '模型', '毛绒玩具', '电子玩具', '拼图']
}

BRANDS = [
    '华为', '小米', '苹果', '三星', '联想', 'OPPO', 'vivo',
    '海尔', '美的', '格力', '索尼', '松下', '飞利浦',
    '耐克', '阿迪达斯', '李宁', '安踏', '特步',
    '无印良品', '宜家', '网易严选', '小米有品'
]


def generate_product_data(size: int = 100000) -> pd.DataFrame:
    """
    生成产品数据集

    Args:
        size: 生成的记录数量

    Returns:
        pandas DataFrame
    """
    print(f"正在生成 {size:,} 条产品记录...")

    data = []
    start_date = datetime.now() - timedelta(days=365)

    for i in range(size):
        if (i + 1) % 10000 == 0:
            print(f"已生成 {i + 1:,} 条记录...")

        # 选择类别和子类别
        category = random.choice(list(CATEGORIES.keys()))
        subcategory = random.choice(CATEGORIES[category])

        # 生成价格
        original_price = round(random.uniform(10, 5000), 2)
        discount = random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50])
        price = round(original_price * (1 - discount / 100), 2)

        # 生成评分和评论数
        rating = round(random.uniform(3.0, 5.0), 1)
        review_count = random.randint(0, 10000)

        # 生成库存
        stock = random.randint(0, 1000)

        # 生成时间戳
        created_at = start_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        updated_at = created_at + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23)
        )

        # 随机引入缺失值（约5%）
        if random.random() < 0.05:
            description = None
        else:
            description = fake.text(max_nb_chars=200)

        if random.random() < 0.03:
            brand = None
        else:
            brand = random.choice(BRANDS)

        # 创建记录
        record = {
            'product_id': f'P{i+1:08d}',
            'name': f'{brand or "未知品牌"} {subcategory} {fake.word()}',
            'category': category,
            'subcategory': subcategory,
            'price': price,
            'original_price': original_price,
            'discount': discount,
            'rating': rating if random.random() > 0.02 else None,
            'review_count': review_count,
            'stock': stock,
            'brand': brand,
            'description': description,
            'created_at': created_at,
            'updated_at': updated_at
        }

        data.append(record)

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 引入一些重复记录（约2%）
    duplicate_count = int(size * 0.02)
    if duplicate_count > 0:
        duplicates = df.sample(n=duplicate_count, random_state=42)
        df = pd.concat([df, duplicates], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✓ 成功生成 {len(df):,} 条记录（包含 {duplicate_count:,} 条重复记录）")

    return df


def save_data(df: pd.DataFrame, output_dir: str = '.'):
    """
    保存数据为多种格式

    Args:
        df: pandas DataFrame
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n正在保存数据...")

    # CSV
    csv_path = output_path / 'products.csv'
    df.to_csv(csv_path, index=False)
    csv_size = csv_path.stat().st_size / (1024 * 1024)
    print(f"✓ CSV: {csv_path} ({csv_size:.2f} MB)")

    # Parquet
    parquet_path = output_path / 'products.parquet'
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)
    print(f"✓ Parquet: {parquet_path} ({parquet_size:.2f} MB)")

    # JSON
    json_path = output_path / 'products.json'
    df.to_json(json_path, orient='records', lines=True)
    json_size = json_path.stat().st_size / (1024 * 1024)
    print(f"✓ JSON: {json_path} ({json_size:.2f} MB)")

    print(f"\n数据统计:")
    print(f"  总记录数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  缺失值: {df.isnull().sum().sum():,}")
    print(f"  重复记录: {df.duplicated().sum():,}")
    print(f"\n类别分布:")
    print(df['category'].value_counts())


def main():
    parser = argparse.ArgumentParser(description='生成示例产品数据集')
    parser.add_argument(
        '--size',
        type=int,
        default=100000,
        help='生成的记录数量（默认: 100000）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='输出目录（默认: 当前目录）'
    )

    args = parser.parse_args()

    # 生成数据
    df = generate_product_data(args.size)

    # 保存数据
    save_data(df, args.output)

    print("\n✓ 数据生成完成！")
    print("\n使用方法:")
    print("  import daft")
    print("  df = daft.read_csv('products.csv')")
    print("  # 或使用 Parquet 格式（推荐）")
    print("  df = daft.read_parquet('products.parquet')")


if __name__ == '__main__':
    main()
