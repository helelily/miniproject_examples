from mrjob.job import MRJob
from mrjob.step import MRStep
import heapq
from lxml import etree
import mwparserfromhell
import math


class MREntropy(MRJob):
    
    def mapper_init_counts(self):
        self.chars = {}

    def mapper_counts(self, _, line):
        try:
            tree = etree.fromstring(line)
            
            for item in tree.findall('.//text'):
                text_extract = item.text
                text_parse = mwparserfromhell.parse(text_extract)
                text = " ".join(" ".join(fragment.value.split()) for fragment in text_parse.filter_text())  
                text_code = text.encode(encoding='UTF-8',errors='ignore').decode(encoding='UTF-8',errors='ignore')
                
                try:
                    for i in xrange(len(text_code)-2):
                        char = text_code[i:i+3]
                        self.chars.setdefault(char, 0)
                        self.chars[char] = self.chars[char] + 1
                        
                except TypeError:
                    pass
                    
        except etree.XMLSyntaxError: 
            pass
    
    def mapper_final_counts(self):
        for char, count in self.chars.iteritems():
            yield char, count
        
    def combiner_counts(self, char, counts):
        yield char, sum(counts)
        
    def reducer_counts(self, char, counts):
        yield None, (char, sum(counts))
    
    def reducer_entropy(self, _, counts):
        big_n = 0
        inner_sum = 0
        for char, count in counts:
            big_n += count
            inner_sum += count * math.log(count, 2)
        
        yield "entropy", math.log(big_n, 2) - (1 / float(big_n)) * float(inner_sum)
                                                             
    def steps(self):
        return [ MRStep(mapper_init=self.mapper_init_counts,
                        mapper=self.mapper_counts,
                        mapper_final=self.mapper_final_counts,
                        combiner=self.combiner_counts,
                        reducer=self.reducer_counts),
                 MRStep(reducer=self.reducer_entropy) ]


if __name__ == '__main__':
    MREntropy.run()
            