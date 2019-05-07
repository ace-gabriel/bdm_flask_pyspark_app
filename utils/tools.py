from pyspark import SparkContext, SparkConf, SQLContext
from random import random
from operator import add
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql import Row
from collections import OrderedDict
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

File = 'data/reduce_input.csv'

def calc_dis(drop, pick):
    geolocator = Nominatim(user_agent="taxi")
    drop_loc, pick_loc = geolocator.geocode(drop), geolocator.geocode(pick)
    dl, pl = (drop_loc.latitude, drop_loc.longitude), (pick_loc.latitude, pick_loc.longitude)
    return geodesic(dl, pl).miles

def parse(num, drop, pick, file=File):

    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    distance = calc_dis(drop, pick)
    # initailize params
    prcp = 0.0413
    snow = 0.0603
    tavg = 47.9109
    tmax = 56.8859
    tmin = 39.8287

    # create new df of input data
    def to_row(d):
        return Row(**OrderedDict(sorted(d.items())))

    df = sc.parallelize([{'PRCP': prcp, 'SNOW': snow, 'TAVG': tavg, \
                          'TMAX': tmax, 'TMIN': tmin, 'passenger_count': float(num), \
                          'trip_distance': float(distance)}]).map(to_row).toDF()

    # assemble to vector
    flist = ['PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN', 'passenger_count', 'trip_distance']
    data2 = df.select(df.PRCP, df.SNOW, df.TAVG, df.TMAX, df.TMIN, df.passenger_count,
                      df.trip_distance)
    assembler = VectorAssembler(inputCols=flist, outputCol='features')
    temp = assembler.transform(data2)
    data = temp.select("features")

    # new model instance
    train = sqlContext.read.format('csv').options(header='true', inferSchema='true').load(file)

    train_data = train.select(train.PRCP, train.SNOW, train.TAVG, train.TMAX, train.TMIN, train.passenger_count,
                              train.trip_distance, train.total_amount.alias('label'))
    train_data = train_data.dropna()
    assembler = VectorAssembler(inputCols=flist, outputCol='features')
    temp = assembler.transform(train_data)
    train_vector = temp.select("features", "label")
    train_vector.show(n=10, truncate=False)

    lr = LinearRegression(featuresCol='features', labelCol='label',
                          maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_vector)
    y_pred = round(lr_model.transform(data).select("prediction").take(1)[0]['prediction'], 2)

    return y_pred
