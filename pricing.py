# -*- coding: utf-8 -*-
# @Author: shayaan
# @Date:   2020-12-05 22:26:28
# @Last Modified by:   shayaan
# @Last Modified time: 2020-12-05 23:05:21

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr


class Pricing(object):
	"""Implementation of the class to Price the product"""
	def __init__(self, start_date, end_date):
		super(Pricing, self).__init__()
		self.start_date = start_date
		self.end_date = end_date
		self.stoxx = pdr.DataReader('^STOXX50E',start=self.start_date,end=self.end_date,data_source='yahoo')['Close'].round(4)
		self.eurusd = pdr.DataReader('EURUSD=X',start=self.start_date,end=self.end_date,data_source='yahoo')['Close']
		self.libor_rate = pdr.DataReader('USD3MTD156N','fred',start=self.start_date,end=self.end_date)

if __name__ == '__main__':
	start_date = "2020-01-01"
	end_date = "2020-12-01"
	test = Pricing(start_date,end_date)
	print(test.libor_rate)