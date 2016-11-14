from mrjob.job import MRJob
from mrjob.step import MRStep
import heapq
from lxml import etree
import mwparserfromhell
import random
import numpy
import itertools


class MRDoubleLinkRank(MRJob):
    def mapper_get_links_per_page(self, _, line):
        try:
            tree = etree.fromstring(line)
            links_per_article = []
            title = tree.find('.//title').text.lower()
            if ':' not in title:
                for item in tree.findall('.//text'):
                    text = item.text
                    try:
                        links = mwparserfromhell.parse(text).filter_wikilinks()
                    except mwparserfromhell.parser.ParserError:
                        pass
                
                    for link in links:
                        try:
                            if ':' in link.title.lower():
                                continue
                            else:
                                links_per_article.append(link.title.lower())
                        except TypeError:
                            pass
                    
                links_per_article = list(set(links_per_article))
                total_unique_links = len(links_per_article) 
                article_weight = 0

                if total_unique_links != 0:
                    article_weight = 1 / (float(total_unique_links) + 10.0)  
        
                yield title, (links_per_article, article_weight, 'C')  #C's

                for link in links_per_article:
                    yield link, (title, article_weight, 'A') #A's
        
        except etree.XMLSyntaxError: 
            pass


    def reducer_get_link_pairs(self, key, values):
        list_of_As = []
        list_of_Cs = []
        for value in values:
            if value[2] == 'A':
                list_of_As.append(value)
            else:
                list_of_Cs.append(value)
        
        list_of_As = sorted(list_of_As, key=lambda x: x[0])
        list_of_Cs = sorted(list_of_Cs, key=lambda x: x[0])
        
        for itemA in list_of_As:
            for itemC in list_of_Cs:
                for link in itemC[0]:
                    if itemA[0] != link:
                        yield ((itemA[0], link), itemA[1] * itemC[1])


    def combiner_count(self, key, values):
        yield (key, sum(values))

    def reducer_sum_weights(self, key, values):
        yield None, (sum(values), key)

    def reducer_find_max_word(self, _, pairs):
        for item in heapq.nlargest(1000,pairs):
            yield item

                     
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_links_per_page, reducer=self.reducer_get_link_pairs),
            MRStep(combiner=self.combiner_count,reducer=self.reducer_sum_weights),
            MRStep(reducer=self.reducer_find_max_word)
        ]


if __name__ == '__main__':
    MRDoubleLinkRank.run()
            
