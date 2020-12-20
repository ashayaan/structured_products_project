# -*- coding: utf-8 -*-
# @Author: shayaan
# @Date:   2020-12-05 22:26:28
# @Last Modified by:   shayaan
# @Last Modified time: 2020-12-20 18:10:58

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from scipy import stats
import numpy as np
import datetime 
import matplotlib.pyplot as plt
np.random.seed(12)

class Pricing(object):
	"""Implementation of the class to Price the product"""
	def __init__(self, start_date:str, end_date:str) -> None:
		'''
		Constructor for the Pricing class
		Input: start_date string, end_date string
		'''
		super(Pricing, self).__init__()
		self.start_date = start_date
		self.end_date = end_date
		self.stoxx = pdr.DataReader('^STOXX50E',start=self.start_date,end=self.end_date,data_source='yahoo').round(4)
		self.eurusd = pdr.DataReader('EURUSD=X',start=self.start_date,end=self.end_date,data_source='yahoo')
		self.libor_rate = pdr.DataReader('USD3MTD156N','fred',start=self.start_date,end=self.end_date)	
		self.risk_free_rate = pdr.DataReader('DTB1YR','fred',start=self.start_date,end=self.end_date)['DTB1YR'][-1]
		self.risk_free_rate /= 100
		self.findReturns()

	def findReturns(self) -> None:
		'''
		Helper function to calculate the return
		for stoxx and eurusd

		Input : instance of class
		output : None
		'''
		self.stoxx['Return'] = (self.stoxx['Close'] - self.stoxx['Close'].shift(1)) / self.stoxx['Close'].shift(1)
		self.stoxx.dropna(inplace=True)
		self.eurusd['Return'] = (self.eurusd['Close'] - self.eurusd['Close'].shift(1)) / self.eurusd['Close'].shift(1)
		self.eurusd.dropna(inplace=True)
		self.libor_rate.dropna(inplace=True)
		self.libor_rate['USD3MTD156N'] /= 100 #as interest rates are quotes in percentage

	def estimateParameters(self) -> None:
		'''
		Function to estimate the parameters for the
		stochastic processes and Vasicek Model
		Input : Class object
		'''
		self.stoxx_std = np.sqrt(np.var(self.stoxx['Return'])*252)
		self.stoxx_mean = (np.mean(self.stoxx['Return'])*252 + (self.stoxx_std**2)/2)
		self.eurusd_std = np.sqrt(np.var(self.eurusd['Return'])*252)
		self.eurusd_mean = (np.mean(self.eurusd['Return']) + (self.eurusd_std**2)/2) 

		st_date = set(self.stoxx.index)
		er_date = set(self.eurusd.index)
		common_date = st_date.intersection(er_date)
		self.rho = stats.pearsonr(self.stoxx[self.stoxx.index.isin(common_date)]['Return'] , self.eurusd[ self.eurusd.index.isin(common_date)]['Return'])[0]

		#Vasicek model estimation
		num = 0
		dem = 0
		r_mean = np.mean(self.libor_rate['USD3MTD156N'])
		print("Mean:{}".format(r_mean))

		for i in range(1,len(self.libor_rate)):
			num += (self.libor_rate.iloc[i]['USD3MTD156N'] - r_mean) * (self.libor_rate.iloc[i-1]['USD3MTD156N'] - r_mean)
			dem += (self.libor_rate.iloc[i]['USD3MTD156N'] - r_mean)**2

		self.a = 1 - (num)/dem
		self.b = r_mean - (1-self.a) * r_mean

		s = 0
		for i in range(1,len(self.libor_rate)):
			s += (self.libor_rate.iloc[i]['USD3MTD156N'] - self.b - (1-self.a)*self.libor_rate.iloc[i-1]['USD3MTD156N'])**2
		
		self.sigma = np.sqrt(s/(len(self.libor_rate)-2))


	def simulateLIBOR(self,T,delta_t,M):
		T = 365 * T
		delta_t = delta_t * 30
		rates = np.zeros(M)

		r = self.libor_rate.iloc[-1]['USD3MTD156N'] 
		for j in range((T - delta_t )):
			weiner_process = np.random.normal(0,1)
			r = self.b + (1-self.a) * r + self.sigma * weiner_process

		return max(r,0)

	def estimatePrice(self, T:int, K:float , K_:float , M:int, forward_rate:float, delta_t=3)->float:
		'''
		Input
		T : number of years as int
		K : strike price float
		K_ : strike price float
		stoxx_0 : The price of eurusd at time of issuance float
		M : time steps of simulation
		forward_rate : forward rate float
		quanto_0 : The starting price of qunato float
		delta_t : look back time in months, by default 3 
		'''

		stoxx_0 = self.stoxx.iloc[-1]['Close']
		eurusd_0 = self.eurusd.iloc[-1]['Close']
		quanto_0 = stoxx_0 * eurusd_0

		W_1 = np.random.normal(0,T,M)
		W_2 = np.random.normal(0,T,M)
		quanto = stoxx_0 * eurusd_0 * np.exp((self.stoxx_mean * self.eurusd_mean - (self.stoxx_std**2 + self.eurusd_std**2)/2)*T + self.stoxx_std*W_1 + self.eurusd_std*W_2 )
		libor = self.simulateLIBOR(T,delta_t,M)
		plt.plot(range(1,M+1),quanto)
		price = max((quanto[-1]/quanto_0 -K ) * (libor/forward_rate - K_),0)
		return price /(1 + self.risk_free_rate)


if __name__ == '__main__':
	start_date = "2020-01-01"
	end_date =  datetime.datetime.now().strftime('%Y-%m-%d')
	test = Pricing(start_date,end_date)
	test.estimateParameters()
	price = []
	nSims = 10
	prices = np.zeros(nSims)
	# 
	for i in range(nSims):
		prices[i] = test.estimatePrice(1, 0.01, 0.01, 100, 0.0022)
	plt.xlabel('Time step')
	plt.ylabel('Qunato porcess')
	plt.title('Qunato Simulation')
	plt.savefig('Simulation')
	print("Price of qunto {}".format(np.mean(prices)))