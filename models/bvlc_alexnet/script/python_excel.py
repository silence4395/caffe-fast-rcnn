#!/usr/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : python_excel.py
## Authors    : zhluo@aries
## Create Time: 2017-09-02:20:26:27
## Description:
## 
##
import sys
import numpy as np
import argparse

from xlwt import Workbook

def parse_args():
    """Parse input arguments"""
    parse = argparse.ArgumentParser(description='TXT to EXCEL')
    parse.add_argument('--source', dest='source', help='source file path')
    parse.add_argument('--output', dest='output', help='output file path', default='result.xls')
    
    args = parse.parse_args()
    
    return args
    
def main(args):
    # create new sheet
    book = Workbook()
    sheet1 = book.add_sheet('sheet 1')

    all_lines = open(args.source, 'r+')
    
    row = 0
    for line in all_lines.readlines():
        if (len(line.split()) > 1):
            source = line.split()
            for i in range(len(line.split())):
                sheet1.write(row, i, source[i])
                sheet1.col(i).width=7000
            row = row + 1
    book.save(args.output)
                 
if __name__=="__main__":
    '''parse arguments'''
    args = parse_args()
    main(args)
