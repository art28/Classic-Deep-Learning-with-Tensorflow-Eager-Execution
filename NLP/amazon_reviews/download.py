import wget

url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'
print("start downloading")
wget.download(url, 'reviews_Beauty_5.json.gz')
print("Done")