import io
import time
from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
from  .models import Place
import pandas as pd
import operator
from statistics import mode, mean

# Create your views here.

ser_his = []

def index(request):
	return render(request, 'index.html')

def search(request):
	location = request.POST["place"]
	location = location.capitalize()
	ser_his.append(location)
	unique = set(ser_his)
	recent = list(unique)
	df = pd.read_csv('search/ekm_town_location_data.csv')
	locs = list(df['Town Name'])
	if location not in locs:
		raise Exception("Sorry, place beyond dataset!")
	else:
		pass
	index = int(df[df['Town Name'] == location].index.values)
	sub_dist = df['Sub District Name'][index]
	sub_place = list(df.loc[df['Sub District Name'] == sub_dist]['Town Name'])

	#Creating object for each sub place
	place_obj = []
	for p in sub_place:
		place_name = str(p)
		p = Place()
		p.name = place_name
		p.bus = df.loc[df['Town Name'] == p.name]['Bus Route Name'].values[0]
		p.railway = df.loc[df['Town Name'] == p.name]['Railway Station Name'].values[0]
		p.college = int(df.loc[df['Town Name'] == p.name]['Total Colleges'].values[0])
		p.health_centre = int(df.loc[df['Town Name'] == p.name]['Total Health Centres'].values[0])
		p.stadium = int(df.loc[df['Town Name'] == p.name]['Total Stadiums'].values[0])
		p.theatre = int(df.loc[df['Town Name'] == p.name]['Total Cinema Theatres'].values[0])
		p.auditorium = int(df.loc[df['Town Name'] == p.name]['Total Auditoriums'].values[0])
		p.home = int(df.loc[df['Town Name'] == p.name]['Number of Households'].values[0])
		area = float(df.loc[df['Town Name'] == p.name]['Area (sq. km.)'].values[0])
		population = int(df.loc[df['Town Name'] == p.name]['Total Population of Town'].values[0])
		p.pop_den = int(population/area)
		density = int((p.pop_den/10000)*100)
		family = int((p.home/100000)*100)
		feature_eq = int((p.HOS*p.health_centre)+(p.COL*p.college)+(p.STD*p.stadium)+(p.THR*p.theatre)+(p.AUD*p.auditorium)+(p.FAM*family)+(p.DEN*density))
		percent = int((feature_eq/700)*100)
		p.rank = percent/10
		place_obj.append(p)
	return render(request, 'search.html', {"location": location, "place": place_obj, "his": recent})
