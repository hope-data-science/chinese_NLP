
library(pacman)
p_load(tidyverse,rio)

setwd("G:\\College_of_big_data\\图书情报分析")

import("层级.xlsx") -> raw

raw %>% as_tibble() %>% select(1) %>% rename(title = `机构规范名`) -> for_embedding

write_csv(for_embedding,"corpus_raw.csv")

##

library(pacman)
p_load(tidyverse,data.table)

#读取文件
fread("classification_corpus_raw.csv",encoding = "UTF-8") %>% 
  as_tibble() %>% 
  mutate(id = 1:n())-> raw


#计算TF-IDF

## 快速分词
p_load(jiebaR)
worker() -> wk

raw %>% 
  mutate(words = map(title,segment,jieba = wk)) %>% 
  select(id,words) -> corpus 

## 计算TF-IDF
corpus %>% 
  unnest() %>% 
  count(id,words) %>% 
  bind_tf_idf(term = words,document = id,n = n) -> corpus_tf_idf

corpus_tf_idf %>% distinct(words)

corpus_tf_idf %>% 
  select(id,words,tf_idf) -> for_future_use

## 尝试筛选词语
string = "大数据学院"

string %>% 
  segment(jiebar = wk) %>% 
  enframe() %>% 
  transmute(words = value) -> string_table

for_future_use %>% 
  inner_join(string_table) %>% 
  group_by(id) %>% 
  summarise(score = sum(tf_idf)) %>% 
  arrange(desc(score)) -> sort_table

sort_table %>% 
  slice(1:5) %>% 
  inner_join(raw,by = "id")

## 函数构造

get_sim = function(string){
  string %>% 
    segment(jiebar = wk) %>% 
    enframe() %>% 
    transmute(words = value) -> string_table
  
  for_future_use %>% 
    inner_join(string_table,by = "words") %>% 
    group_by(id) %>% 
    summarise(score = sum(tf_idf)) %>% 
    arrange(desc(score)) -> sort_table
  
  sort_table %>% 
    slice(1:3) %>% 
    inner_join(raw,by = "id") -> result
  
  ifelse(nrow(result) == 0,
         NA,
         result %>% 
           pull(title) %>%
           str_c(collapse = ","))
}

get_sim("稀奇古怪")

get_sim("大数据")


