#!/usr/bin/env python
# coding: utf-8
import os

files_50m = "50m.txt"

# 1. find files larger than 50M
cmd_50m = f"find -size +50000k > {files_50m}"
os.system(cmd_50m)

# 2. edit lines and save
items = []
line = " "
with open(files_50m, "r") as rf:
    wf = open(".gitignore", "w")
    while len(line):
        line = rf.readline()
        if ".git" not in line.split("/"):
            add_item = line.lstrip("./")
            items.append(add_item)
            wf.write(add_item)
    wf.close()

# 3. verification
[print(item.rstrip("\n")) for item in items]

