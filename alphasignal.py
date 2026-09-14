"""
AlphaSignal - エントリーポイント（互換用ラッパー）
`main.py` への統合に伴う対話式およびCLIインターフェースの呼び出し
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from main import main, run_prediction

if __name__ == '__main__':
    main()
