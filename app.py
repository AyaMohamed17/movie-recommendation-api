from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import os
import json
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()

# 🔹 Initialize FastAPI
app = FastAPI()

# إضافة CORS Middleware (بديل لـ Flask-CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Initialize Firebase
try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_credentials:
        print("FIREBASE_CREDENTIALS environment variable not set.")
        exit(1)
    try:
        # تحليل JSON
        cred_dict = json.loads(firebase_credentials)
        # التأكد من أن private_key في صيغة PEM صحيحة
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].strip()
    except json.JSONDecodeError as e:
        print(f"Invalid FIREBASE_CREDENTIALS format: {e}")
        exit(1)
    # استخدام البيانات مباشرة كـ dictionary
    cred = credentials.Certificate(cred_dict)
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
# Combine category, description, and director for better similarity
df['combined_features'] = (
    df['category'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['crew'].apply(lambda x: x.get('director', '') if isinstance(x, dict) else '').fillna('')
)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# 🔹 Recommendation Functions
def recommend_by_genre(genre):
    # Define common genres to match against
    genre = genre.lower()
    # Map variations of genre names to standard ones
    genre_mapping = {
        "action": "Action",
        "comedy": "Comedy",
        "drama": "Drama",
        "science fiction": "Science Fiction",
        "sci-fi": "Science Fiction",
        "superhero": "Superhero",
        "animation": "Animation"
    }
    standard_genre = genre_mapping.get(genre, genre.capitalize())
    # Case-insensitive search in category
    matches = df[df['category'].str.lower().str.contains(standard_genre.lower(), na=False)]
    return matches.sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

def recommend_by_actor(actor):
    return df[df['cast'].apply(lambda x: actor.lower() in [a.lower() for a in x] if isinstance(x, list) else False)]\
        .sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

def recommend_by_director(director):
    return df[df['crew'].apply(lambda x: director.lower() in x.get('director', '').lower() if isinstance(x, dict) else False)]\
        .sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

def recommend_similar_movies(movie_title):
    if movie_title not in df['name'].values:
        return pd.DataFrame()
    idx = df[df['name'] == movie_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-6:-1][::-1]  # Get top 5 similar movies (excluding the movie itself)
    return df.iloc[indices].sort_values(by=['rating', 'release_date'], ascending=[False, False])

# 🔹 Greetings / Help
def handle_greetings_or_help(user_input):
    greetings = ["hello", "hi", "hey", "hay", "good morning", "good evening", "good afternoon", "مرحبا", "اهلا"]
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
    # Check for greetings or help
    if handle_greetings_or_help(user_input_lower):
        return "greeting_or_help"
    # Check for actor or director (look for "by" keyword)
    if "by" in user_input_lower:
        # Check if the name after "by" matches an actor
        name_after_by = user_input_lower.split("by")[-1].strip()
        all_actors = set(actor for sublist in df['cast'].dropna() for actor in (sublist if isinstance(sublist, list) else []))
        if any(name_after_by in actor.lower() for actor in all_actors):
            return "actor"
        # Check if the name after "by" matches a director
        all_directors = df['crew'].dropna().apply(lambda x: x.get('director') if isinstance(x, dict) else '').dropna().unique()
        if any(name_after_by in director.lower() for director in all_directors):
            return "director"
        # If "by" is present, check if "director" is mentioned
        if "director" in user_input_lower:
            return "director"
        return "actor"  # Default to actor if "by" is present but no match
    # Check for similar movies (look for "like" or "similar to")
    if "like" in user_input_lower or "similar to" in user_input_lower:
        return "similar movie"
    # Check for genre (look for genre names in the input)
    for genre in df['category'].dropna().unique():
        if genre.lower() in user_input_lower:
            return "genre"
    # Check for genre keywords explicitly
    genre_keywords = ["action", "comedy", "drama", "science fiction", "sci-fi", "superhero", "animation"]
    for genre in genre_keywords:
        if genre in user_input_lower:
            return "genre"
    return "unknown"

# 🔹 Final Bot Response
def get_bot_response(user_input):
    # Check for Arabic input
    if contains_arabic(user_input):
        return {
            "bot": "Please use English only for communication.",
            "recommendations": []
        }
    
    # Classify the input
    label = classify_input(user_input)
    user_input_lower = user_input.lower()

    # Handle greetings or help requests
    if label == "greeting_or_help":
        greeting_response = handle_greetings_or_help(user_input_lower)
        return {
            "bot": greeting_response,
            "recommendations": []
        }

    # Handle genre-based recommendations
    if label == "genre":
        genre_keywords = ["action", "comedy", "drama", "science fiction", "sci-fi", "superhero", "animation"]
        matched_genre = None
        for genre in genre_keywords:
            if genre in user_input_lower:
                matched_genre = genre
                break
        # If no keyword matched, try matching against categories in the database
        if not matched_genre:
            for genre in df['category'].dropna().unique():
                if genre.lower() in user_input_lower:
                    matched_genre = genre.lower()
                    break
        if matched_genre:
            recs = recommend_by_genre(matched_genre)
            if not recs.empty:
                formatted_recs = format_recommendations_json(recs)
                return {
                    "bot": f"I recommend movies in the {matched_genre.capitalize()} genre!",
                    "recommendations": formatted_recs,
                    "count": len(formatted_recs)
                }
            return {
                "bot": f"No movies found in the {matched_genre.capitalize()} genre.",
                "recommendations": []
            }
        return {
            "bot": "Please mention a genre to get recommendations.",
            "recommendations": []
        }

    # Handle actor-based recommendations
    elif label == "actor":
        all_actors = set(actor for sublist in df['cast'].dropna() for actor in (sublist if isinstance(sublist, list) else []))
        # Extract the name after "by" if present, otherwise use the whole input
        if "by" in user_input_lower:
            actor_name = user_input_lower.split("by")[-1].strip()
        else:
            actor_name = user_input_lower
        found_actor = None
        for actor in all_actors:
            if actor.lower() in actor_name:
                found_actor = actor
                break
        if found_actor:
            recs = recommend_by_actor(found_actor)
            if not recs.empty:
                formatted_recs = format_recommendations_json(recs)
                return {
                    "bot": f"I recommend movies starring {found_actor}! These are the movies by {found_actor} in our database.",
                    "recommendations": formatted_recs,
                    "count": len(formatted_recs)
                }
            return {
                "bot": f"No movies found for actor {found_actor}.",
                "recommendations": []
            }
        return {
            "bot": f"No movies found for actor {actor_name.capitalize()}.",
            "recommendations": []
        }

    # Handle director-based recommendations
    elif label == "director":
        all_directors = df['crew'].dropna().apply(lambda x: x.get('director') if isinstance(x, dict) else '').dropna().unique()
        # Extract the name after "by" if present, otherwise use the whole input
        if "by" in user_input_lower:
            director_name = user_input_lower.split("by")[-1].strip()
        else:
            director_name = user_input_lower
        # Remove "director" keyword if present
        director_name = director_name.replace("director", "").strip()
        found_director = None
        for director in all_directors:
            if director.lower() in director_name:
                found_director = director
                break
        if found_director:
            recs = recommend_by_director(found_director)
            if not recs.empty:
                formatted_recs = format_recommendations_json(recs)
                return {
                    "bot": f"I recommend movies directed by {found_director}! These are the movies by {found_director} in our database.",
                    "recommendations": formatted_recs,
                    "count": len(formatted_recs)
                }
            return {
                "bot": f"No movies found for director {found_director}.",
                "recommendations": []
            }
        return {
            "bot": f"No movies found for director {director_name.capitalize()}.",
            "recommendations": []
        }

    # Handle similar movie recommendations
    elif label == "similar movie":
        for title in df['name'].dropna().unique():
            if title.lower() in user_input_lower:
                recs = recommend_similar_movies(title)
                if not recs.empty:
                    formatted_recs = format_recommendations_json(recs)
                    return {
                        "bot": f"I recommend movies similar to {title}!",
                        "recommendations": formatted_recs,
                        "count": len(formatted_recs)
                    }
                return {
                    "bot": f"No similar movies found for {title}.",
                    "recommendations": []
                }
        return {
            "bot": "Please mention a valid movie title to find similar ones.",
            "recommendations": []
        }

    # Default response for unrecognized input
    else:
        return {
            "bot": "I'm not sure what you're asking. Try mentioning a genre, actor, director, or a movie title.",
            "recommendations": []
        }

# 🔹 Root Endpoint
@app.get("/")
async def root():
    return {
        "response": {
            "bot": "Welcome to the Movie Recommendation API! Use POST /recommend to get recommendations.",
            "recommendations": []
        }
    }

# 🔹 API Endpoint
@app.post("/recommend")
async def recommend(request: dict):
    user_input = request.get("message", "")
    if not user_input:
        return JSONResponse(content={
            "response": {
                "bot": "Missing message",
                "recommendations": []
            }
        }, status_code=400)
    response = get_bot_response(user_input)
    return {"response": response}

# # 🔹 New Endpoint to Get All Movies
# @app.get("/movies")
# async def get_all_movies():
#     formatted_movies = format_recommendations_json(df)
#     return {
#         "response": {
#             "bot": "Here are all the movies in the database.",
#             "recommendations": formatted_movies,
#             "count": len(formatted_movies)
#         }
#     }

# 🔹 Run Server (For local testing only; Railway uses gunicorn)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)