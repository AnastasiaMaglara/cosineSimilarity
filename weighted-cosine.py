import statistics
from scipy import spatial
import pandas as pd
import numpy as np
import prediction

NUM_OF_USERS = 10
NUM_OF_PARAMETERS = 3
MAX_RATING = 5
MIN_RATING = 1
AVERAGE_UNCOMPUTABLE = (-1000)
COSINE_DISTANCE_UNCOMPUTABLE = np.NaN


weights = [0.7, 0.2, 0.1]

def weightedCosineSim(x, y):
    count = 0
    x_array=[]
    y_array=[]
    for i in x:
        if i in y:
            count += 1
            x_array+=x[i]
            y_array+=y[i]

    weightsOfParameters = []
    for i in range(int(len(x_array)/NUM_OF_PARAMETERS)):
        weightsOfParameters = np.concatenate((weightsOfParameters, weights))

    if count == 0 :
        return COSINE_DISTANCE_UNCOMPUTABLE
    else:
        return 1 - spatial.distance.cosine(scale(x_array), scale(y_array), weightsOfParameters)

def scale(x):
    vector= []
    for i in x:
        vector.append(i-MIN_RATING)
    return vector

def avgRating(ratingDict):
  array = []
  if (len(ratingDict) == 0):
      return AVERAGE_UNCOMPUTABLE

  for a in ratingDict.values():
      array.append(a)
  return [statistics.mean(k) for k in zip(*array)]


ratings = [dict() for x in range(NUM_OF_USERS)]

# read <userid, itemid, rating> triples from dataset
ratings[1][10] = [5, 4, 4];
ratings[1][20] = [3, 4, 3];
ratings[1][30] = [4, 4, 5];
ratings[2][10] = [5, 5, 5];
ratings[2][30] = [4, 5, 3];
ratings[2][100] = [1, 4 ,4];

df = (pd.DataFrame.from_dict(ratings))
df = df.T
print(df)

avgUserRatings = []
for i in range(0, NUM_OF_USERS):
   avgUserRatings.insert(i, avgRating(ratings[i]))
print(avgUserRatings)

sim=pd.DataFrame(columns=df.columns, index=df.columns)
for i in range(0, NUM_OF_USERS):
     for j in range(0, NUM_OF_USERS):
         sim[i][j] = weightedCosineSim(ratings[i],ratings[j])

print(sim)
#print(prediction.predictRatingPearson(ratings, avgUserRatings, sim , NUM_OF_USERS, 2, 20))
