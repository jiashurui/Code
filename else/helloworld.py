#! python
import sys
import csv
from decimal import Decimal, ROUND_HALF_UP

with open('/Users/jiashurui/Desktop/result.csv') as f:
    reader = csv.reader(f)
    fstr = [row for row in reader]
    fmtx = [[sv for sv in row] for row in fstr]
    rmtx = [[Decimal(sv).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) \
             for sv in row[0:50]] for row in fmtx[0:100]]
    print("Your data is read as")
    writer = csv.writer(sys.stdout)
    writer.writerows(rmtx)