from deephaven import learn
from deephaven.TableTools import readCsv

import numpy as np
from sklearn import linear_model

# read in data
insurance = readCsv("/data/examples/insurance/insurance.csv")
# transform variables to categorical
insurance_numeric = insurance.by("region").update("idx = i").dropColumns("region").update("region = idx").dropColumns("idx").ungroup()\
                     .by("sex").update("idx = i").dropColumns("sex").update("sex = idx").dropColumns("idx").ungroup()\
                     .by("smoker").update("idx = i").dropColumns("smoker").update("smoker = idx").dropColumns("idx").ungroup()

# initialize linear model so that it can be used outside of model_func if needed
regr = linear_model.LinearRegression()

# create linear model with scikit learn
def model_func_skl(target, features):

    print("Training model...")
    regr.fit(features, target)

    return regr.predict(features)


def gather(idx, cols):

    rst = np.empty((idx.getSize(), len(cols)), dtype=np.double)
    iter = idx.iterator()
    i = 0

    while(iter.hasNext()):
        it = iter.next()
        j = 0
        for col in cols:
            rst[i,j] = col.get(it)
            j=j+1
        i=i+1

    return rst


def scatter_skl(data, row):
    return np.round(data[row][0], 2)


predicted_skl = learn.learn(
                table = insurance_numeric, model_func = model_func_skl,
                inputs = [learn.Input("charges", gather), learn.Input(["age","bmi","children","region","sex","smoker"], gather)],
                outputs = [learn.Output("predicted_charges", scatter_skl)], batch_size = insurance_numeric.size()
)

####### Deephaven's learn API is highly flexible, so we can use different Python modules to implement the same models ######

import statsmodels.api as sm

# create linear model with statsmodels
def model_func_sm(target, features):

    new_features = sm.add_constant(features)

    print("Training model....")
    model = sm.OLS(target, new_features)
    results = model.fit()

    print(results.summary())

    return results.predict()

# since output is a different shape, scatter must be updated
def scatter_sm(data, row):
    return np.round(data[row], 2)


predicted_sm = learn.learn(
                table = insurance_numeric, model_func = model_func_sm,
                inputs = [learn.Input("charges", gather), learn.Input(["age","bmi","children","region","sex","smoker"], gather)],
                outputs = [learn.Output("predicted_charges", scatter_sm)], batch_size = insurance_numeric.size()
)
