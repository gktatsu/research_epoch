"""
.matファイルに含まれるキー名の一覧を表示するスクリプト
このスクリプトは、指定された.matファイルを読み込み、その中に含まれるキー名を表示します。
また、各キーの値の型や形状、最初の要素のサンプル値も表示します。
このスクリプトは、Pythonのargparseモジュールを使用してコマンドライン引数を解析し、
scipy.ioとh5pyモジュールを使用して.matファイルを読み込みます。
このスクリプトは、Python 3.xで動作します。
このスクリプトは、以下のライブラリに依存しています:
- scipy
- h5py
- argparse
- os
- sys
このスクリプトは、コマンドラインから実行することができます。
使用方法:
python print_mat_keys.py <mat_file>
"""
import os
import sys
import argparse
import scipy.io as sio
import h5py

def print_mat_keys(mat_file):
    """
    .matファイルに含まれるキー名の一覧を表示します
    
    Args:
        mat_file: .matファイルのパス
    """
    try:
        # scipyでmatファイルを読み込む試み
        try:
            data = sio.loadmat(mat_file)
            print(f"\nmatファイル '{mat_file}' のキー:")
            
            # デフォルトで含まれるメタデータキーを除外
            exclude_keys = ['__header__', '__version__', '__globals__']
            for key in data.keys():
                if key not in exclude_keys:
                    value = data[key]
                    shape_str = f"形状: {value.shape}" if hasattr(value, 'shape') else "形状: -"
                    print(f"  - {key} ({shape_str})")
                    
            # 値の型と内容のサンプルを表示
            print("\nキー別の詳細情報:")
            for key in data.keys():
                if key not in exclude_keys:
                    value = data[key]
                    print(f"\n  {key}:")
                    print(f"    型: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"    形状: {value.shape}")
                        if value.size > 0:
                            # NumPy配列の場合、最初の要素をサンプルとして表示
                            print(f"    サンプル値: {value.flatten()[0]}")
                    else:
                        print(f"    値: {value}")
            
            return
        except:
            # scipyで読み込めない場合、h5pyを試す
            pass
        
        # h5pyでmatファイルを読み込む（HDF5形式の新しいmatファイル用）
        with h5py.File(mat_file, 'r') as f:
            print(f"\nmatファイル '{mat_file}' のキー (HDF5形式):")
            
            # 再帰的に全キーを探索
            def print_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shape_str = f"形状: {obj.shape}" if hasattr(obj, 'shape') else "形状: -"
                    print(f"  - {name} ({shape_str})")
            
            # ファイル内のすべてのキーを表示
            f.visititems(print_keys)
    
    except Exception as e:
        print(f"エラー: {e}")

def main():
    parser = argparse.ArgumentParser(description='.matファイルのキー名を表示します')
    parser.add_argument('mat_file', type=str, help='.matファイルのパス')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mat_file):
        print(f"ファイルが見つかりません: {args.mat_file}")
        return
    
    print_mat_keys(args.mat_file)

if __name__ == "__main__":
    main()