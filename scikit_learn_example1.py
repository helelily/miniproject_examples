
# coding: utf-8

# In[ ]:

test1 = {"business_id": "JwUE5GmEO-sH1FuwJgKBlQ", "full_address": "6162 US Highway 51\nDe Forest, WI 53532", "hours": {}, "open": True, "categories": ["Restaurants"], "city": "De Forest", "review_count": 26, "name": "Pine Cone Restaurant", "neighborhoods": [], "longitude": -89.335843999999994, "state": "WI", "stars": 4.0, "latitude": 43.238892999999997, "attributes": {"Take-out": True, "Good For": {"dessert": False, "latenight": False, "lunch": True, "dinner": False, "breakfast": False, "brunch": False}, "Caters": False, "Noise Level": "average", "Takes Reservations": False, "Delivery": False, "Ambience": {"romantic": False, "intimate": False, "touristy": False, "hipster": False, "divey": False, "classy": False, "trendy": False, "upscale": False, "casual": False}, "Parking": {"garage": False, "street": False, "validated": False, "lot": True, "valet": False}, "Has TV": True, "Outdoor Seating": False, "Attire": "casual", "Alcohol": "none", "Waiter Service": True, "Accepts Credit Cards": True, "Good for Kids": True, "Good For Groups": True, "Price Range": 1}, "type": "business"}

test2 = {"business_id": "uGykseHzyS5xAMWoN6YUqA", "full_address": "505 W North St\nDe Forest, WI 53532", "hours": {"Monday": {"close": "22:00", "open": "06:00"}, "Tuesday": {"close": "22:00", "open": "06:00"}, "Friday": {"close": "22:00", "open": "06:00"}, "Wednesday": {"close": "22:00", "open": "06:00"}, "Thursday": {"close": "22:00", "open": "06:00"}, "Sunday": {"close": "21:00", "open": "06:00"}, "Saturday": {"close": "22:00", "open": "06:00"}}, "open": True, "categories": ["American (Traditional)", "Restaurants"], "city": "De Forest", "review_count": 16, "name": "Deforest Family Restaurant", "neighborhoods": [], "longitude": -89.353437, "state": "WI", "stars": 4.0, "latitude": 43.252267000000003, "attributes": {"Take-out": True, "Good For": {"dessert": False, "latenight": False, "lunch": False, "dinner": False, "breakfast": False, "brunch": True}, "Caters": False, "Noise Level": "quiet", "Takes Reservations": False, "Delivery": False, "Parking": {"garage": False, "street": False, "validated": False, "lot": True, "valet": False}, "Has TV": True, "Outdoor Seating": False, "Attire": "casual", "Ambience": {"romantic": False, "intimate": False, "touristy": False, "hipster": False, "divey": False, "classy": False, "trendy": False, "upscale": False, "casual": True}, "Waiter Service": True, "Accepts Credit Cards": True, "Good for Kids": True, "Good For Groups": True, "Price Range": 1}, "type": "business"}

test3 = {"business_id": "LRKJF43s9-3jG9Lgx4zODg", "full_address": "4910 County Rd V\nDe Forest, WI 53532", "hours": {"Monday": {"close": "22:00", "open": "10:30"}, "Tuesday": {"close": "22:00", "open": "10:30"}, "Friday": {"close": "22:00", "open": "10:30"}, "Wednesday": {"close": "22:00", "open": "10:30"}, "Thursday": {"close": "22:00", "open": "10:30"}, "Sunday": {"close": "22:00", "open": "10:30"}, "Saturday": {"close": "22:00", "open": "10:30"}}, "open": True, "categories": ["Food", "Ice Cream & Frozen Yogurt", "Fast Food", "Restaurants"], "city": "De Forest", "review_count": 7, "name": "Culver's", "neighborhoods": [], "longitude": -89.374983, "state": "WI", "stars": 4.5, "latitude": 43.251044999999998, "attributes": {"Take-out": True, "Wi-Fi": "free", "Takes Reservations": False, "Delivery": False, "Parking": {"garage": False, "street": False, "validated": False, "lot": True, "valet": False}, "Wheelchair Accessible": True, "Attire": "casual", "Accepts Credit Cards": True, "Good For Groups": True, "Price Range": 1}, "type": "business"}

test4 = {"business_id": "RgDg-k9S5YD_BaxMckifkg", "full_address": "631 S Main St\nDe Forest, WI 53532", "hours": {"Monday": {"close": "22:00", "open": "11:00"}, "Tuesday": {"close": "22:00", "open": "11:00"}, "Friday": {"close": "22:30", "open": "11:00"}, "Wednesday": {"close": "22:00", "open": "11:00"}, "Thursday": {"close": "22:00", "open": "11:00"}, "Sunday": {"close": "21:00", "open": "16:00"}, "Saturday": {"close": "22:30", "open": "11:00"}}, "open": True, "categories": ["Chinese", "Restaurants"], "city": "De Forest", "review_count": 3, "name": "Chang Jiang Chinese Kitchen", "neighborhoods": [], "longitude": -89.343721700000003, "state": "WI", "stars": 4.0, "latitude": 43.2408748, "attributes": {"Take-out": True, "Has TV": False, "Outdoor Seating": False, "Attire": "casual"}, "type": "business"}


# In[ ]:

import json
import gzip

data = []
with gzip.open('yelp_set.json.gz', 'rb') as f:
    line = json.loads(f.readline())
    data.append(line)
    while line:
        line = json.loads(f.readline())
        data.append(line)

X = data
y = [item['stars'] for item in data]        


# In[ ]:

from sklearn import preprocessing, neighbors, cross_validation, grid_search, base, linear_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer


class AttributeTransformer(base.BaseEstimator, base.TransformerMixin):   
    def __init__(self):
        self.vec = DictVectorizer(sparse=False)
        self.model = linear_model.LassoCV()
        pass
    
    def handle_key_value(self, key, value):
        if type(value) is bool:
            return (key, int(value))
        if type(value) is str or type(value) is unicode:
            return ("_".join([key, value]), 1)
        if type(value) is int:
            return (key, value)

    def handle_dict(self, key, dict_):
        key_value_pairs = []
        for item in dict_:
            new_key = "_".join([key, item])
            pair = self.handle_key_value(new_key, dict_[item])
            key_value_pairs.append(pair)
        return key_value_pairs

    def process_attribs(self, attrib):
        list_ = []
        for item in attrib:
            if type(attrib[item]) is dict:
                items = self.handle_dict(item, attrib[item])
            else:
                items = [self.handle_key_value(item, attrib[item])]
            list_ += items
        return dict(list_)
    
    def handle_formatting(self, X):
        if type(X) is dict:
            attribute_list = [X['attributes']]
        else:
            attribute_list = [item['attributes'] for item in X]
        
        attribute_dicts = []
        for attribute in attribute_list:
            attribute_dict = self.process_attribs(attribute)
            attribute_dicts.append(attribute_dict)
        
        return attribute_dicts
    
    def fit(self, X, y):
        X_form = self.handle_formatting(X)
        self.vec.fit(X_form)
        X_tran = self.vec.transform(X_form)
        self.model.fit(X_tran, y)
        return self

    def transform(self, X): 
        X_form = self.handle_formatting(X)
        X_tran = self.vec.transform(X_form)
        output = self.model.predict(X_tran)
        return output.reshape(len(output),1)


# In[ ]:

class CategoryTransformer(base.BaseEstimator, base.TransformerMixin):   
    def __init__(self):
        self.vec = DictVectorizer(sparse=False)
        self.model = linear_model.LassoCV()
        pass
    
    def handle_formatting(self, X):
        if type(X) is dict:
            category_list = [X['categories']]
        else:
            category_list = [item['categories'] for item in X]
        
        category_dicts = []
        for item in category_list:
            cat_dct = dict((key, 1) for key in item)
            category_dicts.append(cat_dct)
        
        return category_dicts
    
    def fit(self, X, y):
        X_form = self.handle_formatting(X)
        self.vec.fit(X_form)
        X_tran = self.vec.transform(X_form)
        self.model.fit(X_tran, y)
        return self

    def transform(self, X): 
        X_form = self.handle_formatting(X)
        X_tran = self.vec.transform(X_form)
        output = self.model.predict(X_tran)
        return output.reshape(len(output),1)


# In[ ]:

class LocationTransformer(base.BaseEstimator, base.TransformerMixin):   
    def __init__(self):
        self.model = neighbors.KNeighborsRegressor(15)
        pass
    
    def handle_formatting(self, X):
        if type(X) is dict:
            X = [X['longitude'], X['latitude']]
        else:
            columns = []
            for record in X:
                selection = [record['longitude'], record['latitude']]
                columns.append(selection)
            X = columns
        return X
    
    def fit(self, X, y):
        X_form = self.handle_formatting(X)
        self.model.fit(X_form, y)
        return self

    def transform(self, X): 
        X_form = self.handle_formatting(X)
        output = self.model.predict(X_form)
        return output.reshape(len(output),1)


# In[ ]:

import pandas
import numpy

class CityTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.city_map = {}
        self.cities_avg = 0
        
        pass

    def fit(self, X, y):
        directory = [(item['business_id'], item['city'], item['stars']) for item in X]
        df = pandas.DataFrame(directory, columns = ['business_id', 'city', 'stars'])
        self.city_map = df.groupby('city').mean()['stars'].to_dict()
        self.cities_avg = df['stars'].mean()
        return self

    def transform(self, X):
        import numpy
        if type(X) is dict:
            city = X['city']
            X = city
            if X in self.city_map:
                return self.city_map[city]
            else:
                return self.cities_avg
        else:
            columns = []
            for record in X:
                city = record['city']
                if city in self.city_map:
                    addition = self.city_map[city]
                else:
                    addition = self.cities_avg
                columns.append(addition)
            output = numpy.array(columns)
            return output.reshape(len(output),1)


# In[ ]:

from sklearn import pipeline

all_features = pipeline.FeatureUnion([
  ('city', CityTransformer()),
  ('loc', LocationTransformer()),
  ('cats', CategoryTransformer()),
  ('attribs', AttributeTransformer())
  ])
k_union = pipeline.Pipeline([
  ("features", all_features),
  ("linreg", linear_model.RidgeCV(fit_intercept=True))
  ])


# In[ ]:

X = data
y = [item['stars'] for item in data]        

k_union.fit(X,y)


# In[ ]:

city = CityTransformer().fit_transform(X,y)

#k_union.predict([test1])


# In[ ]:

import dill
with open('question5model.p', 'wb') as out_strm: 
    dill.dump(k_union, out_strm)


# In[ ]:



