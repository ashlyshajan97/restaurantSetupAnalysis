import io
import time
from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from  .models import Place, Restaurant
import pandas as pd
from statistics import mode, mean
from sklearn import preprocessing
import pandas as pd
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
# Import label encoder
from sklearn import preprocessing


# Create your views here.

def recommend(Name, cosine_similarities, indices, datasets):
	recommend_restaurant = []
	location = Name
	if Name=='Aluva'or 'aluva'or 'ALUVA':
		Name=0
	elif Name=='Angamaly'or 'angamaly'or 'ANGAMALY':
		Name=1
	elif Name=='Athani'or 'athani'or 'ATHANI':
		Name=2
	elif Name=='Ernakulam'or 'ernamkulam'or 'ERNAKULAM':
		Name=3
	elif Name=='Kakkanad'or 'kakkanad'or 'KAKKANAD':
		Name=4
	elif Name=='Kalady'or 'kalady'or 'KALADY':
		Name=5
	elif Name=='Kalamassery'or 'kalamassery'or 'KALAMASSERY':
		Name=6
	elif Name=='Kochi'or 'kochi'or 'KOCHI':
		Name=7
	elif Name=='Kumbalangy'or 'kumbalangy'or 'KUMBALANGY':
		Name=8
	elif Name=='Maradu'or 'maradu'or 'MARADU':
		Name=9
	elif Name=='Mulanthuruthy'or 'mulanthuruthy'or 'MULANTHURUTHY':
		Name=10
	elif Name=='Mulavukadu'or 'mulavukadu'or 'MULAVUKAD':
		Name=11
	elif Name=='Muvattupuzha'or 'muvattupuzha'or 'MUVATTUPUZHA':
		Name=12
	elif Name=='Nedubassery'or 'nedubassery'or 'NEDUBASSERY':
		Name=13
	elif Name=='Paravur'or 'paravur'or 'PARAVUR':
		Name=14
	elif Name=='Perumbavoor'or 'perumbavoor'or 'PERUMBAVOOR':
		Name=15
	elif Name=='Thrippunithura'or 'thrippunithura'or 'THRIPPUNITHURA':
		Name=16
	elif Name=='Vazhakkala'or 'vazhakkala'or 'VAZHAKKALA':
		Name=17
	# Find the index of the location
	idx = indices[indices == Name].index[0]

	# Find the restaurants with a similar cosine-sim value and order them from biggest number
	score_series = pd.Series(cosine_similarities[idx])
	idx = indices[indices == Name].index[0]

	# Find the restaurants with a similar cosine-sim value and order them from biggest number
	score_series = pd.Series(cosine_similarities[idx])

	# Extract restaurant indexes with a similar cosine-sim value
	top_indexes = list(score_series.iloc[0:].index)

	# Names of the  restaurants
	for each in top_indexes:
		recommend_restaurant.append(list(datasets.index)[each])

	# Creating the new data set to show similar restaurants
	df_new = pd.DataFrame(columns=['Location', 'Cuisines', 'Mean Rating', 'Best Selling', 'Hot Trending'])

	# Create the  similar restaurants with some of their columns
	for each in recommend_restaurant:
		df_new = df_new.append(pd.DataFrame(
			datasets[['Location', 'Cuisines', 'Mean Rating', 'Best Selling', 'Hot Trending']][
				datasets.index == each].sample()))
	result = df_new[df_new["Location"] == location]
	return result

def create_model(location):
	# Importing the datasets
	datasets = pd.read_csv('search/ranking.csv',encoding='latin-1')
	# label_encoder object knows how to understand word labels.
	label_encoder = preprocessing.LabelEncoder()
	# Encode labels
	datasets['Town Name'] = label_encoder.fit_transform(datasets['Town Name'])
	datasets['Town Name'].unique()
	# Computing Mean Rating
	restaurants = list(datasets['Town Name'].unique())
	datasets['Mean Rating'] = 0
	for i in range(len(restaurants)):
		datasets['Mean Rating'][datasets['Town Name'] == restaurants[i]] = datasets['Overall Rating'][
			datasets['Town Name'] == restaurants[i]].mean()
	# Scaling the mean rating values
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(1, 5))
	datasets[['Mean Rating']] = scaler.fit_transform(datasets[['Mean Rating']]).round(2)
	indices = pd.Series(datasets.index)
	# Creating tf-idf matrix
	tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
	tfidf_matrix = tfidf.fit_transform(datasets['review'])
	cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
	return(recommend(location, cosine_similarities, indices, datasets))


ser_his = []
res_his = []

def index(request):
	return render(request, 'index.html')


def search(request):
	location = request.POST["place"]
	location = location.capitalize()
	ser_his.append(location)
	unique = set(ser_his)
	recent = list(unique)
	df = pd.read_csv('search/ekm_town_location_data.csv',encoding='latin-1')
	locs = list(df['Town Name'])
	if location not in locs:
		return render(request, 'search.html', {"location": location,"msg": "Location beyond dataset!","his": recent})
	else:
		pass
	index = int(df[df['Town Name'] == location].index.values)
	sub_dist = df['Sub District Name'][index]
	sub_place = list(df.loc[df['Sub District Name'] == sub_dist]['Town Name'])
	# Creating object for each sub place
	place_obj = []
	for p in sub_place:
		place_name = str(p)
		p = Place()
		result = create_model(place_name)
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
		# density = int((p.pop_den/10000)*100)
		# family = int((p.home/100000)*100)
		# feature_eq = int((p.HOS*p.health_centre)+(p.COL*p.college)+(p.STD*p.stadium)+(p.THR*p.theatre)+(p.AUD*p.auditorium)+(p.FAM*family)+(p.DEN*density))
		# percent = int((feature_eq/700)*100)
		if result.empty:
			p.rank = 0
		else:
			p.rank = float(result['Mean Rating'].unique())
		place_obj.append(p)
	return render(request, 'search.html', {"location": location, "place": place_obj, "his": recent})

def rest(request, loc):
	location = loc
	res_his.append(location)
	unique = set(res_his)
	recent = list(unique)
	df = pd.read_csv('search/restaurants.csv',encoding= 'latin-1')
	rest = list(df.loc[df['Town Name'] == location]['Name'])

	# Creating object for each restaurant
	rest_obj = []
	for r in rest:
		rest_name = str(r)
		r = Restaurant()
		r.name = rest_name
		r.rating = float(df.loc[df['Name'] == r.name]['Rating'].values[0])
		r.best_sale = df.loc[df['Name'] == r.name]['Best Selling'].values[0]
		r.hot_trend = df.loc[df['Name'] == r.name]['Hot Trending'].values[0]
		r.daily_profit = int(df.loc[df['Name'] == r.name]['Average Daily Profit'].values[0])
		r.daily_customers = int(df.loc[df['Name'] == r.name]['Average Daily Customers'].values[0])
		r.crowded_day = df.loc[df['Name'] == r.name]['Most Crowded Day'].values[0]
		rest_obj.append(r)	
	return render(request, 'rest.html', {"location": location, "rest": rest_obj, "his": recent})

def unique_check(l):
	for i in range(len(l)):
		for i1 in range(len(l)):
			if i != i1:
				if l[i] == l[i1]:
					return 1
	return 0

def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 

def insights(request, loc):
	location = loc
	rest_df = pd.read_csv('search/restaurants.csv',encoding= 'latin-1')
	data = pd.read_csv('search/ekm_town_location_data.csv')
	result = create_model(location)
	sub_dist = data.loc[data['Town Name'] == location]['Sub District Name'].values[0]
	dist = data.loc[data['Town Name'] == location]['District Name'].values[0]
	area = data.loc[data['Town Name'] == location]['Area (sq. km.)'].values[0]
	population = data.loc[data['Town Name'] == location]['Total Population of Town'].values[0]
	formatted_time = time.ctime()
	
	# Foodie Zone Status
	cus = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Customers'])
	num = len(cus)
	if (num == 0):
		foodie_zone = "Unknown"
	else:
		foodie = int((sum(cus)/(num*100))*100)
		if (foodie > 0 and foodie <= 300):
			foodie_zone = "Low"
		elif (foodie > 300 and foodie <= 1000):
			foodie_zone = "Medium"
		elif (foodie > 1000 and foodie <= 2000):
			foodie_zone = "High"
		else:
			foodie_zone = "No"
	
	# Predicted Intrinsic Cuisine
	best_sale = list(rest_df.loc[rest_df['Town Name'] == location]['Best Selling'])
	hot_trend = list(rest_df.loc[rest_df['Town Name'] == location]['Hot Trending'])
	dish_data = best_sale + hot_trend
	if not dish_data:
		must_buy_dish = "Unknown"
	elif (unique_check(dish_data) == 0):
		must_buy_dish = "No intrinsic cuisines"
	else:
		must_buy_dish = most_frequent(dish_data)

	# Most Crowded Day
	crowded = list(rest_df.loc[rest_df['Town Name'] == location]['Most Crowded Day'])
	if not crowded:
		most_crowded = "Unknown"
	elif (unique_check(crowded) == 0):
		most_crowded = "No any specific day"
	else:
		most_crowded = most_frequent(crowded)

	# Predicted Risk Level
	profit = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Profit'])
	num = len(profit)
	if (num == 0):
		risk_level = "Unknown"
	else:
		risk = int((sum(profit)/(num*100000))*100)
		if (risk >= 0 and risk <= 5):
			risk_level = "High"
		elif (risk > 5 and risk <= 20):
			risk_level = "Medium"
		elif (risk > 20 and risk <= 30):
			risk_level = "Low"
		else:
			risk_level = "Very Low"

	# Predicted Average Daily Foodies
	foodie_df = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Customers'])
	if not foodie_df:
		avg_foodie = "Unknown"
	else:
		avg_foodie = int(mean(foodie_df))

	# Predicted Average Daily Profit
	profit_df = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Profit'])
	if not profit_df:
		avg_profit = "Unknown"
	else:
		avg_profit = int(mean(profit_df))

	# Types of Restaurant
	if result.empty:
		types = {"types": "No specific types"}
	else:
		types = {"types": list(set(result['Cuisines']))}

	# Best Selling Cuisines
	if result.empty:
		best_selling = {"best": "No specific cuisines"}
	else:
		best_selling = {"best": list(set(result['Best Selling']))}

	# Hot Trending Cuisines
	if result.empty:
		hot_trending = {"hot": "No specific cuisines"}
	else:
		hot_trending = {"hot":(set(result['Hot Trending']))}

	lookup_data = {"formatted_time": formatted_time, "location": location, "sub_dist": sub_dist, "dist": dist, "area": area, "population": population, "foodie_zone": foodie_zone, "must_buy_dish": must_buy_dish, "most_crowded": most_crowded, "risk_level": risk_level, "avg_profit": avg_profit, "avg_foodie": avg_foodie}
	lookup_data.update(types)
	lookup_data.update(best_selling)
	lookup_data.update(hot_trending)
	return render(request, 'lookup.html', lookup_data)

#def lookup(request, loc):
	location = loc
	rest_df = pd.read_csv('search/restaurants.csv')

	#Foodie Zone Status
	cus = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Customers'])
	num = len(cus)
	if (num == 0):
		foodie_zone = "Unknown"
	else:
		foodie = int((sum(cus)/(num*100))*100)
		if (foodie > 0 and foodie <= 30):
			foodie_zone = "Low"
		elif (foodie > 30 and foodie <= 70):
			foodie_zone = "Medium"
		elif (foodie > 70 and foodie <= 100):
			foodie_zone = "High"
		else:
			foodie_zone = "No"
	
	#Predicted Intrinsic Cuisine
	best_sale = list(rest_df.loc[rest_df['Town Name'] == location]['Best Selling'])
	hot_trend = list(rest_df.loc[rest_df['Town Name'] == location]['Hot Trending'])
	dish_data = best_sale + hot_trend
	if not dish_data:
		must_buy_dish = "Unknown"
	elif (unique_check(dish_data) == 0):
		must_buy_dish = "No intrinsic cuisines"
	else:
		must_buy_dish = most_frequent(dish_data)

	#Most Crowded Day
	crowded = list(rest_df.loc[rest_df['Town Name'] == location]['Most Crowded Day'])
	if not crowded:
		most_crowded = "Unknown"
	elif (unique_check(crowded) == 0):
		most_crowded = "No any specific day"
	else:
		most_crowded = most_frequent(crowded)

	#Predicted Risk Level
	profit = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Profit'])
	num = len(profit)
	if (num == 0):
		risk_level = "Unknown"
	else:
		risk = int((sum(profit)/(num*100000))*100)
		if (risk >= 0 and risk <= 5):
			risk_level = "High"
		elif (risk > 5 and risk <= 20):
			risk_level = "Medium"
		elif (risk > 20 and risk <= 30):
			risk_level = "Low"
		else:
			risk_level = "Very Low"

	#Predicted Average Daily Foodies
	foodie_df = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Customers'])
	if not foodie_df:
		avg_foodie = "Unknown"
	else:
		avg_foodie = int(mean(foodie_df))

	#Predicted Average Daily Profit
	profit_df = list(rest_df.loc[rest_df['Town Name'] == location]['Average Daily Profit'])
	if not profit_df:
		avg_profit = "Unknown"
	else:
		avg_profit = int(mean(profit_df))

	#pdf generation
	data = pd.read_csv('search/ekm_town_location_data.csv')
	sub_dist = data.loc[data['Town Name'] == location]['Sub District Name'].values[0]
	dist = data.loc[data['Town Name'] == location]['District Name'].values[0]
	area = data.loc[data['Town Name'] == location]['Area (sq. km.)'].values[0]
	population = data.loc[data['Town Name'] == location]['Total Population of Town'].values[0]

	response = HttpResponse(content_type='application/pdf')
	response['Content-Disposition'] = 'attachment; filename = "%s.pdf"' % location
	doc = SimpleDocTemplate(response,pagesize=letter,
                        rightMargin=72,leftMargin=72,
                        topMargin=72,bottomMargin=18)
	doc.title = '%s' % location
	
	Story=[]
	logo = "search/logo.PNG"
	formatted_time = time.ctime()
	im = Image(logo, 2*inch, 1.85*inch)
	Story.append(im)
	styles=getSampleStyleSheet()
	styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
	
	Story.append(Spacer(1, 18))
	ptext = '<font size="11">%s</font>' % formatted_time
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 18))
	ptext = '<font size="16">%s</font>' % location
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 12))
	ptext = '<font size="11">Sub District Name: %s</font>' % sub_dist.capitalize()
	Story.append(Paragraph(ptext, styles["Italic"]))
	Story.append(Spacer(1, 2))
	ptext = '<font size="11">District Name: %s</font>' % dist.capitalize()
	Story.append(Paragraph(ptext, styles["Italic"]))
	Story.append(Spacer(1, 2))
	ptext = '<font size="11">Area (sq. km.): %s</font>' % area
	Story.append(Paragraph(ptext, styles["Italic"]))
	Story.append(Spacer(1, 2))
	ptext = '<font size="11">Total Population of Town: %s</font>' % population
	Story.append(Paragraph(ptext, styles["Italic"]))

	Story.append(Spacer(1, 18))
	

	ptext = '<font size="14">Insights</font>'
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 12))
	ptext = '<font size="11">Foodies Zone Status: %s</font>' % foodie_zone.capitalize()
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 6))
	ptext = '<font size="11">Predicted Intrinsic Cuisine: %s</font>' % must_buy_dish.capitalize()
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 6))
	ptext = '<font size="11">Most Crowded Day: %s</font>' % most_crowded.capitalize()
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 6))
	ptext = '<font size="11">Predicted Risk Level: %s</font>' % risk_level.capitalize()
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 6))
	ptext = '<font size="11">Predicted Average Daily Foodies: ≈ %s</font>' % avg_foodie
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 6))
	ptext = '<font size="11">Predicted Average Daily Profit: ≈ Rs %s</font>' % avg_profit
	Story.append(Paragraph(ptext, styles["Normal"]))
	Story.append(Spacer(1, 18))

	ptext = '<font size="11">Thank you very much and we look forward to serving you.</font>'
	Story.append(Paragraph(ptext, styles["Justify"]))
	
	Story.append(Spacer(1, 18))
	ptext = '<font size="9">(NB: Only for those who need to start your own restaurant.)</font>'
	Story.append(Paragraph(ptext, styles["Italic"]))
	Story.append(Spacer(1, 18))
	doc.build(Story)
	return response