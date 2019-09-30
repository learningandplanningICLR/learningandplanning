from google.cloud import firestore


def create_firestore_client():
    return firestore.Client()
