#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Encoding Fix
"""

import sys
import os
import codecs

def apply_windows_encoding_fix():
    """Windows環境での文字化け対策を適用する"""
    if sys.platform == 'win32':
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
