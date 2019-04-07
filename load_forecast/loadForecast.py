"""
Neural net implementation of electric load forecasting.
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy.stats import zscore

# NERC6 holidays with inconsistent dates. Created with python holidays package
# years 1990 - 2024
with open('holidays.pickle', 'rb') as f:
	nerc6 = pickle.load(f)

def isHoliday(holiday, df):
	# New years, memorial, independence, labor day, Thanksgiving, Christmas
	m1 = None
	if holiday == "New Year's Day":
		m1 = (df["dates"].dt.month == 1) & (df["dates"].dt.day == 1)
	if holiday == "Independence Day":
		m1 = (df["dates"].dt.month == 7) & (df["dates"].dt.day == 4)
	if holiday == "Christmas Day":
		m1 = (df["dates"].dt.month == 12) & (df["dates"].dt.day == 25)
	m1 = df["dates"].dt.date.isin(nerc6[holiday]) if m1 is None else m1
	m2 = df["dates"].dt.date.isin(nerc6.get(holiday + " (Observed)", []))
	return m1 | m2

def add_noise(m, std):
	noise = np.random.normal(0, std, m.shape[0])
	return m + noise

def makeUsefulDf(df, noise=2.5, hours_prior=24):
	"""
	Turn a dataframe of datetime and load data into a dataframe useful for
	machine learning. Normalize values and turn 
	Features are placed into r_df (return dataframe), creates the following columns

		YEARS SINCE 2000

		LOAD AT THIS TIME DAY BEFORE

		HOUR OF DAY
		- is12AM (0, 1)
		- is1AM (0, 1)
		...
		- is11PM (0, 1)

		DAYS OF THE WEEK
		- isSunday (0, 1)
		- isMonday (0, 1)
		...
		- isSaturday (0, 1)

		MONTHS OF THE YEAR
		- isJanuary (0, 1)
		- isFebruary (0, 1)
		...
		- isDecember (0, 1)

		TEMPERATURE
		- Celcius (normalized from -1 to 1)

		PREVIOUS DAY'S LOAD 
		- 12AM of day previous (normalized from -1 to 1)
		- 1AM of day previous (normalized from -1 to 1)
		...
		- 11PM of day previous (normalized from -1 to 1)

		HOLIDAYS (the nerc6 holidays)
		- isNewYears (0, 1)
		- isMemorialDay (0, 1)
		...
		- is Christmas (0, 1)

	"""

	def _chunks(l, n):
		return [l[i : i + n] for i in range(0, len(l), n)]
	
	if 'dates' not in df.columns:
		df['dates'] = df.apply(
			lambda x: dt(
				int(x['year']), 
				int(x['month']), 
				int(x['day']), 
				int(x['hour'])), 
			axis=1
		)
    
	r_df = pd.DataFrame()
	r_df["load_n"] = zscore(df["load"])
	r_df["years_n"] = zscore(df["dates"].dt.year)

	# fix outliers
	temp = df["tempc"].replace([-9999], np.nan)
	temp.ffill(inplace=True)
	# day-before predictions
	temp_noise = add_noise(temp, noise)
	r_df["temp_n"] = zscore(temp_noise)
	r_df['temp_n^2'] = r_df["temp_n"] ** 2

	# add the value of the load 24hrs before
	r_df["load_prev_n"] = r_df["load_n"].shift(hours_prior)
	r_df["load_prev_n"].bfill(inplace=True)

	# create day of week vector
	r_df["day"] = df["dates"].dt.dayofweek  # 0 is Monday.
	w = ["M", "T", "W", "R", "F", "A", "S"]
	for i, d in enumerate(w):
		r_df[d] = (r_df["day"] == i).astype(int)

		# create hour of day vector
	r_df["hour"] = df["dates"].dt.hour
	d = [("h" + str(i)) for i in range(24)]
	for i, h in enumerate(d):
		r_df[h] = (r_df["hour"] == i).astype(int)

		# create month vector
	r_df["month"] = df["dates"].dt.month
	y = [("m" + str(i)) for i in range(1, 13)]
	for i, m in enumerate(y):
		r_df[m] = (r_df["month"] == i).astype(int)

		# create 'load day before' vector
	n = np.array([val for val in _chunks(list(r_df["load_prev_n"]), 24) for _ in range(24)])
	l = ["l" + str(i) for i in range(24)]
	for i, s in enumerate(l):
		r_df[s] = n[:, i]

		# create holiday booleans
	r_df["isNewYears"] = isHoliday("New Year's Day", df)
	r_df["isMemorialDay"] = isHoliday("Memorial Day", df)
	r_df["isIndependenceDay"] = isHoliday("Independence Day", df)
	r_df["isLaborDay"] = isHoliday("Labor Day", df)
	r_df["isThanksgiving"] = isHoliday("Thanksgiving", df)
	r_df["isChristmas"] = isHoliday("Christmas Day", df)

	m = r_df.drop(["month", "hour", "day", "load_n"], axis=1)
	df = df.drop(['dates'], axis=1)

	return m

def neural_net_predictions(all_X, all_y, EPOCHS=10):
	import tensorflow as tf
	from tensorflow.keras import layers

	X_train, y_train = all_X[:-8760], all_y[:-8760]

	model = tf.keras.Sequential([
		layers.Dense(all_X.shape[1], activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
		layers.Dense(all_X.shape[1], activation=tf.nn.relu),
		layers.Dense(all_X.shape[1], activation=tf.nn.relu),
		layers.Dense(all_X.shape[1], activation=tf.nn.relu),
		layers.Dense(all_X.shape[1], activation=tf.nn.relu),
		layers.Dense(1)
	  ])

	optimizer = tf.keras.optimizers.RMSprop(0.001)

	model.compile(
		loss="mean_squared_error",
		optimizer=optimizer,
		metrics=["mean_absolute_error", "mean_squared_error"],
	)

	early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

	history = model.fit(
		X_train,
		y_train,
		epochs=EPOCHS,
		validation_split=0.2,
		verbose=0,
		callbacks=[early_stop],
	)

	def MAPE(predictions, answers):
		assert len(predictions) == len(answers)
		return sum([abs(x-y)/(y+1e-5) for x, y in zip(predictions, answers)])/len(answers)*100   
	
	predictions = [float(f) for f in model.predict(all_X[-8760:])]
	train = [float(f) for f in model.predict(all_X[:-8760])]
	accuracy = {
		'test': MAPE(predictions, all_y[-8760:]),
		'train': MAPE(train, all_y[:-8760])
	}
	
	return [float(f) for f in model.predict(all_X[-8760:])], accuracy