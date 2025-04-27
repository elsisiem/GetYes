from gradio_client import Client


def sentiment_analysis(prompt):
    client = Client("lloorree/SamLowe-roberta-base-go_emotions")
    result = client.predict(param_0="Hello!!", api_name="/predict")
    return result
