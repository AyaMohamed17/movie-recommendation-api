from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import os
import json


# 🔹 Initialize Flask
app = Flask(__name__)
CORS(app)

# 🔹 Initialize Firebase
# try:
#     cred = credentials.Certificate("service-account.json")
#     firebase_admin.initialize_app(cred)
#     db = firestore.client()
#     movies_ref = db.collection("Movies")
# except Exception as e:
#     print(f"Error initializing Firebase: {e}")
#     exit(1)

try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if firebase_credentials:
        cred_dict = json.loads(firebase_credentials)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    movies_ref = db.collection("Movies")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit(1)

# 🔹 Fetch Movie Data from Firebase
def fetch_movies_from_firebase():
    try:
        print("Fetching movies from Firestore...")
        docs = movies_ref.limit(100).stream()
        movies = [doc.to_dict() for doc in docs]
        print(f"Fetched {len(movies)} movies")
        if not movies:
            print("No movies found in Firestore.")
            return pd.DataFrame()
        return pd.DataFrame(movies)
    except Exception as e:
        print(f"Error fetching movies: {e}")
        return pd.DataFrame()

df = fetch_movies_from_firebase()
if df.empty:
    print("No data loaded from Firestore. Exiting...")
    exit(1)

# 🔹 Clean Data
df['description'] = df['description'].fillna('')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['rating', 'release_date'])

# 🔹 TF-IDF for Similar Movie Recommendations
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# 🔹 Recommendation Functions
def recommend_by_genre(genre):
    return df[df['category'].str.contains(genre, case=False, na=False)].head(5)

def recommend_by_actor(actor):
    return df[df['cast'].apply(lambda x: actor.lower() in [a.lower() for a in x] if isinstance(x, list) else False)].head(5)

def recommend_by_director(director):
    return df[df['crew'].apply(lambda x: director.lower() in x.get('director', '').lower() if isinstance(x, dict) else False)].head(5)

def recommend_similar_movies(movie_title):
    if movie_title not in df['name'].values:
        return pd.DataFrame()
    idx = df[df['name'] == movie_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-6:-1][::-1]
    return df.iloc[indices]

# 🔹 Greetings / Help
def handle_greetings_or_help(user_input):
    greetings = ["hello", "hi", "hey", "hay", "good morning", "good evening", "good afternoon"]
    help_requests = ["help", "can you help", "what can you do", "tell me about movies", "recommend a movie"]
    user_input_lower = user_input.lower()
    for greeting in greetings:
        if greeting in user_input_lower:
            return "Hello! How can I assist you today? Feel free to ask for movie recommendations or anything else."
    for help_request in help_requests:
        if help_request in user_input_lower:
            return "I can recommend movies based on genre, actor, director, or similar movies. Just let me know what you're looking for!"
    return None

# 🔹 Check for Arabic Text
def contains_arabic(text):
    return bool(re.search(r'[؀-ۿ]', text))

# 🔹 Plain Text Formatter (for Postman)
def format_recommendations_json(recommendations):
    if not recommendations.empty:
        recommendations = recommendations.sort_values(by=['rating', 'release_date'], ascending=[False, False])
        formatted_recommendations = []
        for _, row in recommendations.iterrows():
            try:
                year = row['release_date'].year if pd.notnull(row['release_date']) else 'Unknown'
            except AttributeError:
                year = 'Unknown'
            try:
                cast_list = row['cast']
                if isinstance(cast_list, str):
                    cast_list = ast.literal_eval(cast_list)
            except Exception:
                cast_list = []
            try:
                director = row['crew']['director'] if isinstance(row['crew'], dict) and 'director' in row['crew'] else 'Unknown'
            except Exception:
                director = 'Unknown'
            formatted_recommendations.append({
                "movie_title": row['name'],
                "genre": row['category'],
                "imdb_score": row['rating'],
                "release_year": year,
                "cast": cast_list,
                "director": director
            })
        return formatted_recommendations
    return []

# 🔹 Classify Input (Updated to use keyword-based rules)
def classify_input(user_input):
    user_input_lower = user_input.lower()
    # Check for actor or director (look for "by" keyword)
    if " by " in user_input_lower:
        # Check if the name after "by" matches an actor
        name_after_by = user_input_lower.split(" by ")[-1].strip()
        all_actors = set(actor for sublist in df['cast'].dropna() for actor in (sublist if isinstance(sublist, list) else []))
        if any(name_after_by in actor.lower() for actor in all_actors):
            return "actor"
        # Check if the name after "by" matches a director
        all_directors = df['crew'].dropna().apply(lambda x: x.get('director') if isinstance(x, dict) else '').dropna().unique()
        if any(name_after_by in director.lower() for director in all_directors):
            return "director"
        # If "by" is present but no match, assume it's an actor or director request
        return "actor"  # Default to actor if "by" is present but no match
    # Check for similar movies (look for "like" or "similar to")
    if "like" in user_input_lower or "similar to" in user_input_lower:
        return "similar movie"
    # Check for genre (look for genre names in the input)
    for genre in df['category'].dropna().unique():
        if genre.lower() in user_input_lower:
            return "genre"
    # Default to genre if no other match
    return "genre"

# 🔹 Final Bot Response
def get_bot_response(user_input):
    if contains_arabic(user_input):
        return {"bot": "Please use English only for communication."}
    greeting_or_help_response = handle_greetings_or_help(user_input)
    if greeting_or_help_response:
        return {"bot": greeting_or_help_response}
    label = classify_input(user_input)
    user_input_lower = user_input.lower()
    if label == "genre":
        for genre in df['category'].dropna().unique():
            if genre.lower() in user_input_lower:
                recs = recommend_by_genre(genre)
                if not recs.empty:
                    return {
                        "bot": f"I recommend movies in the {genre} genre!",
                        "recommendations": format_recommendations_json(recs)
                    }
                return {"bot": f"No movies found in the {genre} genre."}
        return {"bot": "Please mention a genre to get recommendations."}
    elif label == "actor":
        all_actors = set(actor for sublist in df['cast'].dropna() for actor in (sublist if isinstance(sublist, list) else []))
        # Extract the name after "by" if present, otherwise use the whole input
        if " by " in user_input_lower:
            actor_name = user_input.split(" by ")[-1].strip()
        else:
            actor_name = user_input
        found_actor = None
        for actor in all_actors:
            if actor.lower() in actor_name.lower():
                found_actor = actor
                break
        if found_actor:
            recs = recommend_by_actor(found_actor)
            if not recs.empty:
                return {
                    "bot": f"I recommend movies starring {found_actor}! These are the movies by {found_actor} in our database.",
                    "recommendations": format_recommendations_json(recs)
                }
            return {"bot": f"No movies found for actor {found_actor}."}
        return {"bot": f"No movies found for actor {actor_name}."}
    elif label == "director":
        all_directors = df['crew'].dropna().apply(lambda x: x.get('director') if isinstance(x, dict) else '').dropna().unique()
        # Extract the name after "by" if present, otherwise use the whole input
        if " by " in user_input_lower:
            director_name = user_input.split(" by ")[-1].strip()
        else:
            director_name = user_input
        found_director = None
        for director in all_directors:
            if director.lower() in director_name.lower():
                found_director = director
                break
        if found_director:
            recs = recommend_by_director(found_director)
            if not recs.empty:
                return {
                    "bot": f"I recommend movies directed by {found_director}! These are the movies by {found_director} in our database.",
                    "recommendations": format_recommendations_json(recs)
                }
            return {"bot": f"No movies found for director {found_director}."}
        return {"bot": f"No movies found for director {director_name}."}
    elif label == "similar movie":
        for title in df['name'].dropna().unique():
            if title.lower() in user_input_lower:
                recs = recommend_similar_movies(title)
                if not recs.empty:
                    return {
                        "bot": f"I recommend movies similar to {title}!",
                        "recommendations": format_recommendations_json(recs)
                    }
                return {"bot": f"No similar movies found for {title}."}
        return {"bot": "Please mention a valid movie title to find similar ones."}
    else:
        return {"bot": "I'm not sure what you're asking. Try mentioning a genre, actor, director, or a movie title."}

# 🔹 API Endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Missing message"}), 400
    response = get_bot_response(user_input)
    return jsonify({"response": response})

# 🔹 Run Server
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
