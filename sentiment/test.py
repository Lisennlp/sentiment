



class Doc:
    def __init__(self):
        self.field = {}
    def add(self, field, content):
        self.field[field] = content
    def get(self, field):
        return self.field[field]


#  建索引
# from doc import Doc
import jieba
import math
import json


class Indexer:
    inverted = {}  # 记录词所在文档及词频
    idf = {}  # 词的逆文档频率
    id_doc = {}  # 文档与词的对应关系

    def __init__(self, file_path):
        self.doc_list = []
        self.index_writer(file_path)

    def index_writer(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                print(line)
                line = json.loads(line)
                key, title = line['content'], line['id']
                doc = {}
                doc.update({'text': key})
                doc.update({'title': title})
                # doc.update({'link': link})
                self.doc_list.append(doc)
        self.index()

    def index(self):
        doc_num = len(self.doc_list)  # 文档总数
        for doc in self.doc_list:
            key = doc.get('key')
            # 正排
            self.id_doc[key] = doc

            # 倒排
            term_list = list(jieba.cut_for_search(doc.get('title')))  # 分词
            for t in term_list:
                if t in self.inverted:
                    if key not in self.inverted[t]:
                        self.inverted[t][key] = 1
                    else:
                        self.inverted[t][key] += 1
                else:
                    self.inverted[t] = {key: 1}

        for t in self.inverted:
            self.idf[t] = math.log10(doc_num / len(self.inverted[t]))

        print(f'self.inverted) = {self.inverted}')
        print("inverted terms:%d" % len(self.inverted))
        print(f'idf = {self.idf}')
        print("index done")

# from index import Indexer
import jieba
import operator
import math

"""
搜索
返回结果：(相关问题,相似度)列表
搜索步骤：
    1.分词
    2.计算tf-idf,找出候选doc
    3.对文档排序
"""


class Searcher:

    def __init__(self, index):
        self.index = index

    def search(self, query):
        term_list = []
        query = query.split()
        for entry in query:
            # 分词
            term_list.extend(jieba.cut_for_search(entry))

        # 计算tf-idf,找出候选doc
        tf_idf = {}
        print(term_list)
        for term in term_list:
            print(f'term = {term}')
            print(f'term_list = {term_list}')

            if term in self.index.inverted:

                for doc_id, fre in self.index.inverted[term].items():
                    print(f'fre = {fre}, doc_id = {doc_id}')
                    if doc_id in tf_idf:

                        tf_idf[doc_id] += (1 + math.log10(fre)) * self.index.idf[term]
                    else:
                        tf_idf[doc_id] = (1 + math.log10(fre)) * self.index.idf[term]
                    print(f'tf-idf = {tf_idf}')
        # 排序
        sorted_doc = sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)

        res = [(score, self.index.id_doc[doc_id]) for doc_id, score in sorted_doc]
        return res




if __name__ == '__main__':
    print("index")
    doc_index = Indexer("/Users/lisen/Desktop/test.txt")
    res = Searcher(doc_index).search('白癜风')
    print(res)



