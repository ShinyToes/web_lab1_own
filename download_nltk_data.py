#!/usr/bin/env python3
"""
NLTK 数据包下载和修复脚本
解决 WordNet 等数据包的下载和损坏问题
"""

import os
import sys
import nltk
import zipfile
import shutil
import requests
from pathlib import Path
import time


class NLTKDataDownloader:
    """NLTK 数据包下载器"""
    
    def __init__(self):
        self.nltk_data_dir = self._get_nltk_data_dir()
        self.required_packages = [
            'punkt',
            'punkt_tab', 
            'wordnet',
            'omw-1.4',
            'stopwords',
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng'
        ]
        
    def _get_nltk_data_dir(self):
        """获取 NLTK 数据目录"""
        try:
            # 尝试找到现有的 NLTK 数据目录
            for path in nltk.data.path:
                if os.path.exists(path):
                    return path
        except:
            pass
        
        # 使用默认目录
        home_dir = os.path.expanduser("~")
        return os.path.join(home_dir, "nltk_data")
    
    def _download_package(self, package_name, max_retries=3):
        """下载单个数据包"""
        print(f"正在下载 {package_name}...")
        
        for attempt in range(max_retries):
            try:
                print(f"  尝试 {attempt + 1}/{max_retries}")
                nltk.download(package_name, quiet=False, force=True)
                
                # 验证下载
                if self._verify_package(package_name):
                    print(f"  OK {package_name} 下载成功")
                    return True
                else:
                    print(f"  ERROR {package_name} 下载验证失败")
                    
            except Exception as e:
                print(f"  ERROR 下载失败: {e}")
                if attempt < max_retries - 1:
                    print(f"  等待 5 秒后重试...")
                    time.sleep(5)
        
        return False
    
    def _verify_package(self, package_name):
        """验证数据包是否正确下载"""
        try:
            # 根据包名检查不同的路径
            if package_name == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package_name == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif package_name == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif package_name == 'omw-1.4':
                nltk.data.find('corpora/omw-1.4')
            elif package_name == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif package_name == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif package_name == 'averaged_perceptron_tagger_eng':
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            
            return True
        except:
            return False
    
    def _clean_corrupted_files(self):
        """清理损坏的数据包文件"""
        print("检查并清理损坏的数据包...")
        
        corpora_dir = os.path.join(self.nltk_data_dir, 'corpora')
        if os.path.exists(corpora_dir):
            for file_name in os.listdir(corpora_dir):
                if file_name.endswith('.zip'):
                    file_path = os.path.join(corpora_dir, file_name)
                    try:
                        # 尝试打开 zip 文件
                        with zipfile.ZipFile(file_path, 'r') as zip_file:
                            zip_file.testzip()
                        print(f"  OK {file_name} 正常")
                    except zipfile.BadZipFile:
                        print(f"  ERROR {file_name} 损坏，删除...")
                        os.remove(file_path)
                    except Exception as e:
                        print(f"  WARNING {file_name} 检查失败: {e}")
    
    def _manual_download_wordnet(self):
        """手动下载 WordNet 数据包"""
        print("尝试手动下载 WordNet...")
        
        try:
            # WordNet 下载 URL
            url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
            
            corpora_dir = os.path.join(self.nltk_data_dir, 'corpora')
            os.makedirs(corpora_dir, exist_ok=True)
            
            wordnet_zip_path = os.path.join(corpora_dir, 'wordnet.zip')
            
            # 下载文件
            print("  正在下载 WordNet zip 文件...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(wordnet_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("  正在解压 WordNet...")
            with zipfile.ZipFile(wordnet_zip_path, 'r') as zip_file:
                zip_file.extractall(corpora_dir)
            
            # 删除 zip 文件
            os.remove(wordnet_zip_path)
            
            print("  OK WordNet 手动下载成功")
            return True
            
        except Exception as e:
            print(f"  ERROR WordNet 手动下载失败: {e}")
            return False
    
    def _manual_download_omw(self):
        """手动下载 OMW 数据包"""
        print("尝试手动下载 OMW-1.4...")
        
        try:
            # OMW 下载 URL
            url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip"
            
            corpora_dir = os.path.join(self.nltk_data_dir, 'corpora')
            os.makedirs(corpora_dir, exist_ok=True)
            
            omw_zip_path = os.path.join(corpora_dir, 'omw-1.4.zip')
            
            # 下载文件
            print("  正在下载 OMW-1.4 zip 文件...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(omw_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("  正在解压 OMW-1.4...")
            with zipfile.ZipFile(omw_zip_path, 'r') as zip_file:
                zip_file.extractall(corpora_dir)
            
            # 删除 zip 文件
            os.remove(omw_zip_path)
            
            print("  OK OMW-1.4 手动下载成功")
            return True
            
        except Exception as e:
            print(f"  ERROR OMW-1.4 手动下载失败: {e}")
            return False
    
    def download_all_packages(self):
        """下载所有必需的数据包"""
        print("=" * 60)
        print("NLTK 数据包下载器")
        print("=" * 60)
        print(f"NLTK 数据目录: {self.nltk_data_dir}")
        print()
        
        # 确保数据目录存在
        os.makedirs(self.nltk_data_dir, exist_ok=True)
        
        # 清理损坏的文件
        self._clean_corrupted_files()
        print()
        
        success_count = 0
        failed_packages = []
        
        for package in self.required_packages:
            print(f"处理数据包: {package}")
            
            # 检查是否已存在
            if self._verify_package(package):
                print(f"  OK {package} 已存在且正常")
                success_count += 1
                continue
            
            # 尝试下载
            if self._download_package(package):
                success_count += 1
            else:
                # 对于关键包，尝试手动下载
                if package == 'wordnet':
                    if self._manual_download_wordnet():
                        success_count += 1
                        continue
                elif package == 'omw-1.4':
                    if self._manual_download_omw():
                        success_count += 1
                        continue
                
                failed_packages.append(package)
            
            print()
        
        # 最终验证
        print("=" * 60)
        print("最终验证")
        print("=" * 60)
        
        for package in self.required_packages:
            if self._verify_package(package):
                print(f"OK {package}")
            else:
                print(f"ERROR {package}")
        
        print()
        print(f"成功下载: {success_count}/{len(self.required_packages)} 个数据包")
        
        if failed_packages:
            print(f"失败的包: {', '.join(failed_packages)}")
            return False
        else:
            print("所有数据包下载成功！")
            return True


def main():
    """主函数"""
    try:
        downloader = NLTKDataDownloader()
        success = downloader.download_all_packages()
        
        if success:
            print("\nOK NLTK 数据包下载完成！")
            print("现在可以运行事件文本处理器了。")
            return 0
        else:
            print("\nERROR 部分数据包下载失败")
            print("请检查网络连接或手动下载失败的数据包")
            return 1
            
    except Exception as e:
        print(f"\nERROR 下载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
