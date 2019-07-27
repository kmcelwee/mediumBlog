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
	machine learning. Normalize values.
	"""
	def _isHoliday(holiday, df):
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
	
	if 'dates' not in df.columns:
		df['dates'] = df.apply(lambda x: dt(int(x['year']), int(x['month']), int(x['day']), int(x['hour'])), axis=1)

	r_df = pd.DataFrame()
	
	# LOAD
	r_df["load_n"] = zscore(df["load"])
	r_df["load_prev_n"] = r_df["load_n"].shift(hours_prior)
	r_df["load_prev_n"].bfill(inplace=True)
	
	# LOAD PREV
	def _chunks(l, n):
		return [l[i : i + n] for i in range(0, len(l), n)]
	n = np.array([val for val in _chunks(list(r_df["load_n"]), 24) for _ in range(24)])
	l = ["l" + str(i) for i in range(24)]
	for i, s in enumerate(l):
		r_df[s] = n[:, i]
		r_df[s] = r_df[s].shift(hours_prior)
		r_df[s] = r_df[s].bfill()
	r_df.drop(['load_n'], axis=1, inplace=True)
	
	# DATE
	r_df["years_n"] = zscore(df["dates"].dt.year)
	r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.hour, prefix='hour')], axis=1)
	r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.dayofweek, prefix='day')], axis=1)
	r_df = pd.concat([r_df, pd.get_dummies(df.dates.dt.month, prefix='month')], axis=1)
	for holiday in ["New Year's Day", "Memorial Day", "Independence Day", "Labor Day", "Thanksgiving", "Christmas Day"]:
		r_df[holiday] = _isHoliday(holiday, df)

	# TEMP
	temp_noise = df['tempc'] + np.random.normal(0, noise, df.shape[0])
	r_df["temp_n"] = zscore(temp_noise)
	r_df['temp_n^2'] = zscore([x*x for x in temp_noise])

	return r_df

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

	optimizer = tf.keras.optimizers.RMSprop(0.0001)

	model.compile(
		loss="mean_squared_error",
		optimizer=optimizer,
		metrics=["mean_absolute_error", "mean_squared_error"],
	)

	early_stop = tf.keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=20)

	history = model.fit(
		X_train,
		y_train,
		epochs=EPOCHS,
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