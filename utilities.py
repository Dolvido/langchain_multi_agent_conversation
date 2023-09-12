
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import spacy
import pandas as pd

# Load the pre-trained word vectors from spaCy
nlp = spacy.load("en_core_web_sm")

# Dictionary mapping topic IDs to sets of keywords
topic_keywords_dict = {
    1: {"technology", "AI", "artificial intelligence", "machine learning", "innovation"},
    2: {"environment", "conservation", "climate change", "global warming", "sustainability"},
    3: {"movies", "cinema", "film", "blockbuster", "Hollywood"}
}

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import spacy
import pandas as pd

# Load the pre-trained word vectors from spaCy
nlp = spacy.load("en_core_web_sm")

# Dictionary mapping topic IDs to sets of keywords
topic_keywords_dict = {
    1: {"technology", "AI", "artificial intelligence", "machine learning", "innovation"},
    2: {"environment", "conservation", "climate change", "global warming", "sustainability"},
    3: {"movies", "cinema", "film", "blockbuster", "Hollywood"}
}

def calculate_relevance_score(response, user_query):
    # Preprocessing: get the vector representation of the query and response
    query_vector = nlp(user_query).vector
    response_vector = nlp(response).vector

    # Check if the vectors are non-zero
    if np.any(query_vector) and np.any(response_vector):
        # Step 1: Calculate cosine similarity
        cosine_sim_score = cosine_similarity([query_vector], [response_vector])[0][0]
    else:
        # Set a default value if either vector is zero
        cosine_sim_score = 0.0

    # Step 2: Calculate a score based on the response length
    length_score = len(response.split()) / 100.0

    # Step 3: Calculate a score based on the number of named entities
    doc = nlp(response)
    ner_score = len(doc.ents) / 10.0
            
    # Combining the scores using a weighted sum (weights w1, w2, and w3 will be fine-tuned)
    w1, w2, w3 = 0.5, 0.3, 0.2
    score = w1 * cosine_sim_score + w2 * length_score + w3 * ner_score
            
    return score

def self_evaluating_agent(response, user_query):
    score = calculate_relevance_score(response, user_query)
    
    if score >= 0.1:  # This threshold would be determined through testing and optimization
        return response
    else:
        return "I'm sorry, it seems I went off-topic. Let's refocus."

def user_user_collaborative_filtering(df, user_id, num_recommendations):
    # Create a user-item matrix where rows represent users and columns represent topics
    user_item_matrix = df.pivot_table(index='User_ID', columns='Topic_ID', values='Rating')
        
    # Calculate cosine similarity between users
    user_similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))
        
    # Find the top N similar users (excluding the user itself)
    target_user_index = user_id - 1  # Adjusting for 0-based index
    similar_users = user_similarity_matrix[target_user_index].argsort()[::-1][1:num_recommendations+1]
        
    # Get the list of topics rated by the target user
    target_user_rated_topics = set(df[df['User_ID'] == user_id]['Topic_ID'].tolist())
        
    # Recommend topics rated highly by similar users but not yet rated by the target user
    recommended_topics = set()
    for similar_user_index in similar_users:
        similar_user_id = similar_user_index + 1  # Adjusting for 0-based index
        similar_user_high_rated_topics = set(df[(df['User_ID'] == similar_user_id) & (df['Rating'] >= 4)]['Topic_ID'].tolist())
            
        recommended_topics = recommended_topics.union(similar_user_high_rated_topics.difference(target_user_rated_topics))
            
        # If we have enough recommendations, break the loop
        if len(recommended_topics) >= num_recommendations:
            break
        
    return list(recommended_topics)[:num_recommendations]

def item_item_collaborative_filtering(df, user_id, num_recommendations):
    # Create an item-item matrix where rows represent topics and columns represent users
    item_user_matrix = df.pivot_table(index='Topic_ID', columns='User_ID', values='Rating')
        
    # Calculate cosine similarity between topics
    item_similarity_matrix = cosine_similarity(item_user_matrix.fillna(0))
        
    # Get the list of topics rated highly by the target user
    target_user_high_rated_topics = set(df[(df['User_ID'] == user_id) & (df['Rating'] >= 4)]['Topic_ID'].tolist())
        
    # Get the list of all topics
    all_topics = set(df['Topic_ID'].unique())
        
    # Recommend topics similar to the ones liked by the user but not yet rated by the user
    recommended_topics = set()
    for liked_topic in target_user_high_rated_topics:
        # Find the most similar topics to the liked topic
        similar_topics = item_similarity_matrix[liked_topic - 1].argsort()[::-1][1:num_recommendations+1]
        similar_topics = set(similar_topics + 1)  # Adjusting for 0-based index
            
        # Recommend topics similar to the liked topics but not yet rated by the user
        recommended_topics = recommended_topics.union(similar_topics.difference(target_user_high_rated_topics).difference(all_topics))
            
        # If we have enough recommendations, break the loop
        if len(recommended_topics) >= num_recommendations:
            break
        
    return list(recommended_topics)[:num_recommendations]

def recommend_popular_topics(df, num_recommendations):
    # Calculate the average rating for each topic
    topic_avg_rating = df.groupby('Topic_ID')['Rating'].mean()
        
    # Get the top N topics with the highest average rating
    top_n_topics = topic_avg_rating.nlargest(num_recommendations).index.tolist()
        
    return top_n_topics

def recommend_based_on_content(conversation_text, num_recommendations):
    # Convert the sets of keywords into single strings
    topic_keywords_strings = [" ".join(keywords) for keywords in topic_keywords_dict.values()]
        
    # Vectorize the conversation text and the topic keywords
    vectorizer = CountVectorizer().fit_transform([conversation_text] + topic_keywords_strings)
    vectors = vectorizer.toarray()
        
    # Calculate cosine similarity between the conversation text and each topic's keywords
    cosine_matrix = cosine_similarity(vectors)
        
    # Get the similarity scores for each topic
    topic_similarity_scores = cosine_matrix[0, 1:]
        
    # Get the top N most similar topics
    recommended_topic_ids = topic_similarity_scores.argsort()[::-1][:num_recommendations] + 1  # Adjusting for 0-based index
        
    return recommended_topic_ids.tolist()

def recommend_topics(conversation_history, num_recommendations):
    # Get the conversation text
    conversation_text = " ".join([input_query for _, input_query, _ in conversation_history])

    # Recommend topics based on the conversation text
    recommended_topic_ids = recommend_based_on_content(conversation_text, num_recommendations)

    # Recommend topics based on the user's previous ratings
    # (This could be done using user_user_collaborative_filtering() or item_item_collaborative_filtering())
    # ...

    # Combine the recommendations from the two methods
    recommended_topics = []
    for topic_id in recommended_topic_ids:
        if topic_id not in recommended_topics:
            recommended_topics.append(topic_id)

    return recommended_topics
