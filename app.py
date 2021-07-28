import os
from flask import Flask, render_template, request
import urllib3
import requests
import json

from werkzeug.utils import secure_filename

import openvino_infer

app = Flask(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#Variables for the
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'mov'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST', 'GET'])
def main():
    # recipes[i] corresponds to images[i] corresponds to missed_ing_nums_[i] ... etc.
    ingredients = set()
    recipes = []
    images = []
    missed_ingredient_numbers = []
    # keys = recipe names, values = missed ingdts per recipe
    missed_ingredient_names = {}

    if request.method == 'POST':

        #Download the files that are uploaded in the frontend
        file = request.files['file']
        if file.filename == '':
            print('No file selected for inferring')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('Image successfully uploaded and displayed below')

        num_recipes_to_show = 5
        ignore_pantry = True
        sorting_priority = 1
        ingredients.update(openvino_infer.infer())

        dir = 'Uploads'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        recipe_json = CallAPI(ingredients, num_recipes_to_show, ignore_pantry, sorting_priority)

        for i in range(len(recipe_json)):
            for ingredient in recipe_json[i]['missedIngredients']:

                if recipe_json[i]['title'] not in missed_ingredient_names:
                    missed_ingredient_names[recipe_json[i]['title']] = []

                missed_ingredient_names[recipe_json[i]['title']].append(ingredient['name'])

            recipes.append(recipe_json[i]['title'])
            images.append(recipe_json[i]['image'])
            missed_ingredient_numbers.append(recipe_json[i]['missedIngredientCount'])

        return render_template('app.html', ingredients=list(ingredients), recipes=recipes, images=images,
        missed_ingredient_numbers = missed_ingredient_numbers, missed_ingredient_names=missed_ingredient_names)

    return render_template('app.html', ingredients=["Upload ingredients to get recommendations!"],
    recipes=["Upload ingredients to get recommendations!"], images=images, 
    missed_ingredient_numbers = missed_ingredient_numbers, missed_ingredient_names=missed_ingredient_names)


"""
Send a GET request to Spoonacular API, and return recipes that use the specified ingredients
@:param ingredients, a list of ingredients outputted by the image classification model
@:param num_recipes_to_show, user-specified number of recipes to return
@:param ignore_pantry, bool value whether to ignore pantry ingredients (salt, water, etc) or not
@:param sorting_priority, whether to maximize used ingredients (1) or minimize missing ingredients (2) first
@:return recipe_json, a json formatted list of recipes and associated metadata
"""
def CallAPI(ingredients, num_recipes_to_show, ignore_pantry, sorting_priority):
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/findByIngredients"

    querystring = {"ingredients": ingredients,
                   "number": num_recipes_to_show,
                   "ignorePantry": ignore_pantry,
                   "ranking": sorting_priority
                   }

    headers = {
        'x-rapidapi-key': "82ef15bdc3msh893d3386c0b40d6p1939bajsn7d7255d714f2",
        'x-rapidapi-host': "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    #print(response.text)
    recipe_json = json.loads(response.text)
    return recipe_json

if __name__ == "__main__":
    app.run(debug=True)
