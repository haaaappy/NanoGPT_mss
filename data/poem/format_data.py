# -*- coding: utf-8 -*-
import glob
import json
datas_json=glob.glob("./tang-poetry/poet*.json") #1匹配所有唐诗json文件
 
for data_json in datas_json[:]: #2处理匹配的每一个文件
    with open(data_json,"r",encoding="utf-8") as f:
        ts_data =json.load(f)
        for each_ts in ts_data[:]: #3处理文件中每段数据，只要五言诗和2句的
            paragraphs_list =each_ts["paragraphs"]
            if len(paragraphs_list) == 2 and len(paragraphs_list[0])==12 and len(paragraphs_list[1]) == 12:
                with open("tang_poet.txt","a",encoding="utf-8") as f2:
                    f2.write("".join(paragraphs_list))
                    f2.write("\n")
 
f =open("tang_poet.txt","r",encoding="utf-8")
print(len(f.readlines()))