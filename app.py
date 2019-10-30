import flask
from flask import render_template, request
import pickle
from my_functions import create_random_input, prcess_input
import pandas as pd


app = flask.Flask(__name__, template_folder='templates')




		
@app.route('/', methods=['POST', 'GET'])
def my_form_post():
	if request.method=='GET':
		return render_template('main_zero.html')

	if request.method=='POST':
		if 'generate' in request.form:
			with open('data.pickle', 'rb') as f:
				global data
				data = pickle.load(f)

			data_r = []

			inputs = create_random_input(data['names'], data['languages'], data['genres'])
			for col in inputs.columns:
				data_r.append(inputs[col].values)
			
			global data_tg
			data_tg = data_r.copy()



			return render_template('main.html', data_r=data_r)

		elif 'process' in request.form:	
			
			data_f = list(request.form.to_dict().values())
			
			languages = data_f[0]
			app_purchase = data_f[1]
			price = data_f[2]
			name = data_f[3]
			age_rating = data_f[4]
			size = data_f[5]
			genres = data_f[6]
			original_date = data_f[7]
			current_release = data_f[8]

			inp = pd.DataFrame({'Languages':[','.join(languages)], 'In-app Purchases':[app_purchase], 'Price':[price], 'Name':[name], 'Age Rating':[age_rating], 'Size':[size], 
					  'Genres':[','.join(genres)],	'Original Release Date':['11/07/'+str(original_date)],	'Current Version Release Date':['22/07/'+str(current_release)]})


			final_df = prcess_input(inp, data['languages'], data['genres'], data['age_ratings'], data['vectorizer'], data['svd'], data['yeo'], data['scaler'], data['yeo2'], data['scaler2'], data['a_map'], data['b_map'], data['best_features'])

			with open('model.pickle', 'rb') as f:
				model = pickle.load(f)

			result = model.predict(final_df)


			if result == 0:
				result = 'The App is bad'
			else: result = 'The App is good'	
			return render_template('main.html', data_r=data_tg, result=result)


if __name__ == '__main__':

	app.run()
