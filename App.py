from flask import Flask, request, jsonify, render_template 
import joblib 
import numpy as np 
 
app = Flask(__name__) 
 
# Load trained model 
model = joblib.load('diet_recommender_rf.pkl') 
 
 
# Diet and food mappings with calories 
diet_foods = { 
    "Keto": [ 
        {"name": "Fatty fish (salmon, sardines)", "calories": 208}, 
        {"name": "Avocado", "calories": 160}, 
        {"name": "Nuts and nut butters", "calories": 180}, 
        {"name": "Coconut oil", "calories": 117}, 
        {"name": "Extra virgin olive oil", "calories": 119}, 
        {"name": "Cheese", "calories": 113}, 
        {"name": "Eggs", "calories": 78}, 
        {"name": "Meat and poultry", "calories": 250} 
    ], 
    "Vegan": [ 
        {"name": "Tofu and tempeh", "calories": 145}, 
        {"name": "Legumes (beans, lentils)", "calories": 230}, 
        {"name": "Whole grains", "calories": 216}, 
        {"name": "Fruits and vegetables", "calories": 60}, 
        {"name": "Nuts and seeds", "calories": 180}, 
        {"name": "Plant-based milk", "calories": 90}, 
        {"name": "Nutritional yeast", "calories": 60} 
    ], 
    "Mediterranean": [ 
        {"name": "Fruits and vegetables", "calories": 60}, 
        {"name": "Whole grains", "calories": 216}, 
        {"name": "Legumes", "calories": 230}, 
        {"name": "Fish and seafood", "calories": 200}, 
        {"name": "Olive oil", "calories": 119}, 
        {"name": "Nuts and seeds", "calories": 180}, 
        {"name": "Herbs and spices", "calories": 10} 
    ],
    "Low-Carb": [ 
        {"name": "Lean meats", "calories": 165}, 
        {"name": "Green leafy vegetables", "calories": 25}, 
        {"name": "Cheese", "calories": 113}, 
        {"name": "Eggs", "calories": 78}, 
        {"name": "Avocados", "calories": 160}, 
        {"name": "Nuts and seeds", "calories": 180}, 
        {"name": "Olive oil", "calories": 119} 
    ], 
    "High-Protein": [ 
        {"name": "Chicken breast", "calories": 165}, 
        {"name": "Eggs", "calories": 78}, 
        {"name": "Greek yogurt", "calories": 100}, 
        {"name": "Lentils", "calories": 230}, 
        {"name": "Quinoa", "calories": 222}, 
        {"name": "Tuna", "calories": 132}, 
        {"name": "Cottage cheese", "calories": 98} 
    ] 
} 
 
def contains_nonveg(food_name): 
    nonveg_keywords = ['fish', 'meat', 'chicken', 'tuna', 'poultry', 'egg', 'seafood'] 
    return any(word in food_name.lower() for word in nonveg_keywords) 
 
def calculate_bmr(gender, weight, height, age): 
    if gender == "male": 
        return 10 * weight + 6.25 * height - 5 * age + 5 
    else: 
        return 10 * weight + 6.25 * height - 5 * age - 161 
 
def total_calories(bmr, activity_level): 

    activity_multipliers = { 
        1: 1.2, 
        2: 1.375, 
        3: 1.55, 
        4: 1.725, 
        5: 1.9 
    } 
    return bmr * activity_multipliers.get(activity_level, 1.2) 
 
@app.route('/') 
def home(): 
    return render_template('index.html') 
 
@app.route('/recommend', methods=['POST']) 
def recommend(): 
    try: 
        data = request.json 
 
        age = data['age'] 
        bmi = data['bmi'] 
        activity_level = data['activity_level'] 
        vegetarian = data['vegetarian'] 
        gender = data.get('gender', 'female')  # default to female if missing 
        weight = data.get('weight') 
        height = data.get('height') 
 
        # Calculate BMR and total calories 
        if weight and height: 
            bmr = calculate_bmr(gender, weight, height, age) 
            calories_needed = total_calories(bmr, activity_level) 
        else: 
            bmr = None 
            calories_needed = None 
 
        # Predict diet type 
        features = np.array([ 
            age, 
            bmi, 
            activity_level, 
            vegetarian 
        ]).reshape(1, -1) 
 
        prediction = model.predict(features)[0] 
 
        # Filter foods 
        foods = diet_foods.get(prediction, []) 
        if vegetarian == 1: 
            foods = [f for f in foods if not contains_nonveg(f['name'])] 
 
        return jsonify({ 
            'recommended_diet': prediction, 
            'foods': foods, 
            'bmr': round(bmr, 2) if bmr else None, 
            'calories_needed': round(calories_needed, 2) if calories_needed else None 
        }) 
 
    except Exception as e: 
        return jsonify({'error': str(e)}), 400 
 
if __name__ == '__main__': 
    app.run(debug=True) 
