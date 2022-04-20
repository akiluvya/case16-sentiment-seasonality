from pymongo import MongoClient
import configs

client = MongoClient(configs.mongo_url)
db = client[configs.db_name]
collection_name = db[configs.db_collection]

def insert_tweet_data(data):
    try:
        result = collection_name.insert_one(data)
        return {"result": result, "success": True}
    except Exception as e:
        return {"result": e, "success": False}


def update_tweet_link(data, tweet_id):
    try:
        result = collection_name.update( { '_id': tweet_id }, { '$set': data}, no_cursor_timeout=True )
        return {"result": result, "success": True}
    except Exception as e:
        return {"result": e, "success": False}


def get_unprocessed_tweets():
    try:
        filters = {"processed": False}
        fields = { "_id": 1, "processed": 1}
        result = collection_name.find(filters, fields)
        return {"result": result, "success": True}
    except Exception as e:
        return {"result": e, "success": False}


def get_unfetched_tweets():
    try:
        filters = {"scrapped": False}
        fields = { "_id": 1}
        result = collection_name.find(filters, fields)
        x = [y["_id"] for y in result]
        return {"result": x, "success": True}
    except Exception as e:
        return {"result": e, "success": False}


def update_tweet(data, tweet_id):
    try:
        result = collection_name.replace_one({"_id" : tweet_id}, data, upsert=True)
        print(result)
        return {"result": result, "success": True}
    except Exception as e:
        return {"result": e, "success": False}

