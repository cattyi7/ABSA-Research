# -*- coding: utf-8 -*-
"""
数据集加载模块（已升级支持 Aspect-Based Sentiment Analysis）
支持：
- 中文整句分类：ChnSentiCorp
- 英文整句分类：IMDb
- 英文Aspect分类：SemEval 2014 Task 4 (Laptop / Restaurant)
"""

import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from ..utils.config import Config
from ..utils.text_processor import TextProcessor


class DatasetLoader:
    def __init__(self, language: str = "english", task_type: str = "sentence"):
        self.language = language
        self.task_type = task_type  # sentence / aspect
        self.dataset_name = Config.DATASETS[language]
        self.text_processor = TextProcessor(language)

        self.kaggle_sources = {
            "chinese": {
                "dataset_name": "kaggleyxz/chnsenticorp",
                "url": "https://www.kaggle.com/datasets/kaggleyxz/chnsenticorp/download",
                "file_name": "chnsenticorp.csv",
                "text_column": "text",
                "label_column": "label"
            },
            "english": {
                "dataset_name": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
                "url": "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download",
                "file_name": "IMDB Dataset.csv",
                "text_column": "review",
                "label_column": "sentiment"
            }
        }
        Config.create_directories()

    # -------------------------------------------------------------------------
    # 专门加载 SemEval2014 Aspect 数据集
    # -------------------------------------------------------------------------
    def load_semeval_aspect_dataset(self) -> Optional[Dataset]:
        semeval_dir = Config.DATASETS_DIR / "semeval"
        if not semeval_dir.exists():
            print(f"❌ SemEval 目录不存在: {semeval_dir}")
            return None

        target_file = semeval_dir / "Laptop_Train_v2.csv"
        if not target_file.exists():
            target_file = semeval_dir / "Restaurants_Train_v2.csv"
        if not target_file.exists():
            print("❌ 未找到 Laptop_Train_v2.csv 或 Restaurants_Train_v2.csv")
            return None

        df = pd.read_csv(target_file)
        df = df.rename(columns={
            "Sentence": "text",
            "Aspect Term": "aspect",
            "polarity": "label"
        })

        df = df.dropna(subset=["text", "aspect", "label"])
        df["label"] = df["label"].map({"positive": 1, "negative": 0}).dropna()
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        return Dataset.from_pandas(df[["text", "aspect", "label"]], preserve_index=False)

    def download_from_huggingface(self, cache_dir: str = None) -> Optional[Dataset]:
        if cache_dir is None:
            cache_dir = str(Config.DATASETS_DIR)

        print(f"正在从Hugging Face下载 {self.language} 数据集: {self.dataset_name}")

        try:
            if self.language == "chinese":
                dataset = load_dataset(self.dataset_name, cache_dir=cache_dir, trust_remote_code=True)
                if 'train' in dataset:
                    dataset = dataset['train']
                else:
                    dataset = dataset[list(dataset.keys())[0]]
            else:
                dataset = load_dataset(self.dataset_name, cache_dir=cache_dir, trust_remote_code=True)
                train_data = dataset['train']
                test_data = dataset['test']

                combined_texts = train_data['text'] + test_data['text']
                combined_labels = train_data['label'] + test_data['label']

                dataset = Dataset.from_dict({
                    'text': combined_texts,
                    'label': combined_labels
                })

            print(f"Hugging Face数据集下载完成，共 {len(dataset)} 条记录")
            return dataset

        except Exception as e:
            print(f"从Hugging Face下载数据集失败: {str(e)}")
            return None

    def download_from_kaggle(self, cache_dir: str = None) -> Optional[Dataset]:
        if cache_dir is None:
            cache_dir = str(Config.DATASETS_DIR)

        kaggle_config = self.kaggle_sources.get(self.language)
        if not kaggle_config:
            print(f"不支持的语言类型: {self.language}")
            return None

        print(f"正在从Kaggle下载 {self.language} 数据集: {kaggle_config['dataset_name']}")

        try:
            try:
                import kaggle
                print("使用Kaggle官方API下载...")
                kaggle.api.dataset_download_files(
                    kaggle_config['dataset_name'],
                    path=cache_dir,
                    unzip=True
                )
                csv_files = list(Path(cache_dir).glob("*.csv"))
                if not csv_files:
                    print("未找到CSV数据文件")
                    return None
                csv_file = csv_files[0]
                print(f"读取数据文件: {csv_file}")

            except ImportError:
                print("未安装kaggle包，尝试直接下载...")
                return self._download_kaggle_direct(cache_dir, kaggle_config)

        except Exception as e:
            print(f"Kaggle API下载失败: {str(e)}")
            return self._download_kaggle_direct(cache_dir, kaggle_config)

        try:
            df = pd.read_csv(csv_file)
            if kaggle_config['text_column'] in df.columns and kaggle_config['label_column'] in df.columns:
                df = df.rename(columns={
                    kaggle_config['text_column']: 'text',
                    kaggle_config['label_column']: 'label'
                })
            else:
                print(f"未找到期望的列: {kaggle_config['text_column']}, {kaggle_config['label_column']}")
                return None

            if self.language == "chinese":
                df['label'] = df['label'].map({'positive': 1, 'negative': 0, 1: 1, 0: 0})
            else:
                df['label'] = df['label'].map({'positive': 1, 'negative': 0, 1: 1, 0: 0})

            df = df.dropna(subset=['text', 'label'])
            dataset = Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
            print(f"Kaggle数据集处理完成，共 {len(dataset)} 条记录")
            return dataset

        except Exception as e:
            print(f"处理Kaggle数据集失败: {str(e)}")
            return None

    def _download_kaggle_direct(self, cache_dir: str, kaggle_config: Dict) -> Optional[Dataset]:
        try:
            fallback_urls = {
                "chinese": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
                "english": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            }

            url = fallback_urls.get(self.language)
            if not url:
                print(f"无可用的备用数据源")
                return None

            print(f"从备用源下载数据: {url}")

            if self.language == "chinese":
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                csv_path = Path(cache_dir) / "chinese_sentiment.csv"
                with open(csv_path, 'wb') as f:
                    f.write(response.content)

                df = pd.read_csv(csv_path, encoding='utf-8')
                if 'review' in df.columns and 'label' in df.columns:
                    df = df.rename(columns={'review': 'text'})
                elif len(df.columns) >= 2:
                    df.columns = ['text', 'label'] + list(df.columns[2:])

                df = df.dropna(subset=['text', 'label'])
                df['label'] = df['label'].astype(int)
                dataset = Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
                return dataset
            else:
                print("英文备用数据源下载较复杂，建议手动下载或使用Kaggle API")
                return None

        except Exception as e:
            print(f"备用数据源下载失败: {str(e)}")
            return None

    def download_dataset(self, cache_dir: str = None) -> Dataset:
        if cache_dir is None:
            cache_dir = str(Config.DATASETS_DIR)

        # ========== 【新增】Aspect 模式优先加载 SemEval ==========
        if self.task_type == "aspect" and self.language == "english":
            data = self.load_semeval_aspect_dataset()
            if data is not None:
                return data

        dataset = self.download_from_huggingface(cache_dir)
        if dataset is not None:
            return dataset

        print("Hugging Face下载失败，尝试使用Kaggle备用数据源...")
        dataset = self.download_from_kaggle(cache_dir)
        if dataset is not None:
            return dataset

        raise Exception(f"无法从任何数据源下载 {self.language} 数据集。")

    # -------------------------------------------------------------------------
    # 【升级】预处理：支持 aspect 字段
    # -------------------------------------------------------------------------
    def preprocess_dataset(self, dataset: Dataset, max_samples: int = None) -> Dataset:
        print("正在预处理数据集...")
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        def preprocess_function(examples):
            if 'text' in examples:
                texts = examples['text']
            elif 'review' in examples:
                texts = examples['review']
            else:
                for key, value in examples.items():
                    if isinstance(value[0], str) and key != 'label':
                        texts = value
                        break
                else:
                    raise ValueError("未找到文本字段")

            processed_texts = [self.text_processor.clean_text(t) if t else "" for t in texts]
            output = {"text": processed_texts, "label": examples["label"]}

            # ========== 【新增】保留 aspect ==========
            if "aspect" in dataset.column_names:
                output["aspect"] = examples["aspect"]

            return output

        keep_cols = ["text", "label"]
        if "aspect" in dataset.column_names:
            keep_cols.append("aspect")

        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=[c for c in dataset.column_names if c not in keep_cols]
        )
        processed_dataset = processed_dataset.filter(lambda x: len(x["text"].strip()) > 0)
        print(f"预处理完成，剩余 {len(processed_dataset)} 条有效记录")
        return processed_dataset

    def split_dataset(self, dataset: Dataset, test_size: float = 0.1, val_size: float = 0.1) -> Tuple[
        Dataset, Dataset, Dataset]:
        print(f"正在划分数据集，测试集比例: {test_size}, 验证集比例: {val_size}")
        df = dataset.to_pandas()
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=42,
                                            stratify=train_val_df['label'])

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        print(f"数据集划分完成:")
        print(f"  训练集: {len(train_dataset)} 条")
        print(f"  验证集: {len(val_dataset)} 条")
        print(f"  测试集: {len(test_dataset)} 条")
        return train_dataset, val_dataset, test_dataset

    def get_processed_data(self, max_samples: int = None) -> Tuple[Dataset, Dataset, Dataset]:
        raw_dataset = self.download_dataset()
        processed_dataset = self.preprocess_dataset(raw_dataset, max_samples)
        train_data, val_data, test_data = self.split_dataset(processed_dataset)
        return train_data, val_data, test_data

    def check_dataset_exists(self, cache_dir: str = None) -> bool:
        if cache_dir is None:
            cache_dir = str(Config.DATASETS_DIR)
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return False

        if self.task_type == "aspect":
            sem_dir = cache_path / "semeval"
            if not sem_dir.exists():
                return False
            has_laptop = (sem_dir / "Laptop_Train_v2.csv").exists()
            has_rest = (sem_dir / "Restaurants_Train_v2.csv").exists()
            return has_laptop or has_rest

        if self.language == "chinese":
            seamew_dirs = list(cache_path.glob("**/seamew*")) + list(cache_path.glob("**/ChnSentiCorp*"))
            for d in seamew_dirs:
                fs = list(d.rglob("*.arrow")) + list(d.rglob("*.parquet")) + list(d.rglob("dataset_info.json"))
                if len(fs) > 0:
                    return True
        elif self.language == "english":
            imdb_dirs = list(cache_path.glob("**/imdb*")) + list(cache_path.glob("**/IMDb*"))
            for d in imdb_dirs:
                fs = list(d.rglob("*.arrow")) + list(d.rglob("*.parquet")) + list(d.rglob("dataset_info.json"))
                if len(fs) > 0:
                    return True

        kaggle_config = self.kaggle_sources.get(self.language)
        if kaggle_config:
            csv_files = list(cache_path.glob(f"**/{kaggle_config['file_name']}"))
            if not csv_files:
                csv_files = list(cache_path.glob("**/*.csv"))
            for f in csv_files:
                try:
                    df = pd.read_csv(f)
                    if len(df) > 0 and kaggle_config['text_column'] in df.columns and kaggle_config[
                        'label_column'] in df.columns:
                        return True
                except:
                    continue

        tsv_files = list(cache_path.glob("**/*.tsv"))
        for f in tsv_files:
            try:
                df = pd.read_csv(f, sep='\t')
                if 'label' in df.columns and 'text_a' in df.columns and len(df) > 0:
                    return True
                elif kaggle_config and len(df) > 0 and kaggle_config['text_column'] in df.columns and kaggle_config[
                    'label_column'] in df.columns:
                    return True
            except:
                continue
        return False

    def load_existing_dataset(self, cache_dir: str = None) -> Optional[Dataset]:
        if cache_dir is None:
            cache_dir = str(Config.DATASETS_DIR)

        if self.task_type == "aspect" and self.language == "english":
            return self.load_semeval_aspect_dataset()

        print(f"正在加载已存在的 {self.language} 数据集...")
        try:
            dataset = load_dataset(self.dataset_name, cache_dir=cache_dir, trust_remote_code=True)
            if self.language == "chinese":
                if 'train' in dataset:
                    dataset = dataset['train']
                else:
                    dataset = dataset[list(dataset.keys())[0]]
            else:
                train_data = dataset['train']
                test_data = dataset['test']
                combined_texts = train_data['text'] + test_data['text']
                combined_labels = train_data['label'] + test_data['label']
                dataset = Dataset.from_dict({'text': combined_texts, 'label': combined_labels})

            if dataset is not None and len(dataset) > 0:
                print(f"从Hugging Face缓存加载数据集成功，共 {len(dataset)} 条记录")
                return dataset
        except Exception as e:
            print(f"从Hugging Face缓存加载失败: {str(e)}")

        cache_path = Path(cache_dir)
        kaggle_config = self.kaggle_sources.get(self.language)
        if kaggle_config:
            print("尝试从本地CSV文件加载...")
            csv_files = list(cache_path.glob(f"**/{kaggle_config['file_name']}"))
            if not csv_files:
                csv_files = list(cache_path.glob("**/*.csv"))
            for f in csv_files:
                try:
                    df = pd.read_csv(f)
                    if kaggle_config['text_column'] in df.columns and kaggle_config['label_column'] in df.columns:
                        df = df.rename(columns={kaggle_config['text_column']: 'text',
                                                kaggle_config['label_column']: 'label'})
                        df['label'] = df['label'].map({'positive': 1, 'negative': 0, 1: 1, 0: 0})
                        df = df.dropna(subset=['text', 'label'])
                        if len(df) > 0:
                            return Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
                except:
                    continue

        print("尝试从TSV文件加载...")
        for f in list(cache_path.glob("**/*.tsv")):
            try:
                df = pd.read_csv(f, sep='\t')
                if 'label' in df.columns and 'text_a' in df.columns:
                    df = df.rename(columns={'text_a': 'text'}).dropna(subset=['text', 'label'])
                    df['label'] = df['label'].astype(int)
                    if len(df) > 0:
                        return Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
            except:
                continue
        print("所有加载方式都失败了")
        return None

    def get_or_download_data(self, max_samples: int = None) -> Tuple[Dataset, Dataset, Dataset]:
        if self.check_dataset_exists():
            print(f"检测到已存在的 {self.language} 数据集，直接加载...")
            raw = self.load_existing_dataset()
            if raw is not None:
                processed = self.preprocess_dataset(raw, max_samples)
                return self.split_dataset(processed)
        print(f"未找到已存在的数据集，开始下载...")
        return self.get_processed_data(max_samples)
    
    # ===================== 【导出划分后的CSV】 =====================
    def split_and_save_semeval(self, output_dir: Path):
        """
        先划分 训练/验证/测试 并保存成3个CSV，只对训练集增强，彻底避免泄露！
        """
        #  加载原始纯数据
        df = pd.read_csv(Config.DATASETS_DIR / "semeval" / "Laptop_Train_v2.csv")
        unique_ids = df["id"].unique()
        train_ids, test_ids = train_test_split(unique_ids, test_size=0.1, random_state=42)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

        # 根据 ID 分组取数据
        train = df[df["id"].isin(train_ids)]
        val   = df[df["id"].isin(val_ids)]
        test  = df[df["id"].isin(test_ids)]

        # 保存
        output_dir.mkdir(exist_ok=True)
        train.to_csv(output_dir / "train_original.csv", index=False, encoding="utf-8")
        val.to_csv(output_dir / "val_original.csv", index=False, encoding="utf-8")
        test.to_csv(output_dir / "test_original.csv", index=False, encoding="utf-8")

        print("✅ 按句子ID切分完成（无任何泄露）：")
        print(f"训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")