import statistics
from scipy import spatial
import pandas as pd
import prediction

NUM_OF_USERS = 100
MAX_RATING = 5
MIN_RATING = 1
AVERAGE_UNCOMPUTABLE = (-1000)

def cosineSim(x, y):
    return 1 - spatial.distance.cosine(scale(x), scale(y))

def scale(x):
    vector= []
    for i in x:
        vector.append(i-MIN_RATING)
    return vector

def avgRating(ratingDict):
  if (len(ratingDict) == 0):
    return AVERAGE_UNCOMPUTABLE
  return statistics.mean(ratingDict[k] for k in ratingDict)


ratings = [dict() for x in range(NUM_OF_USERS)]

# read <userid, itemid, rating> triples from dataset
ratings[1][10] = 2;
ratings[1][20] = 1;
ratings[1][30] = 5;
ratings[2][10] = 2;
ratings[2][30] = 4;
ratings[2][100] = 5;

avgUserRatings = []
for i in range(0, NUM_OF_USERS):
  avgUserRatings.insert(i, avgRating(ratings[i]))
print(avgUserRatings)

df = (pd.DataFrame.from_dict(ratings))
df = df.T
sim = pd.DataFrame(df.corr(method=cosineSim))
print(sim)
print(prediction.predictRatingPearson(ratings, avgUserRatings, sim , NUM_OF_USERS, 2, 20))