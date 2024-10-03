from flask import Flask, request, jsonify
from setfit import SetFitModel
from functools import lru_cache

app = Flask(__name__)

# Modify the model loading function to use the Hugging Face token and model name
@lru_cache(maxsize=None)
def load_intent_model():
    model = SetFitModel.from_pretrained("ali170506/chab", 
                                        token="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")
    return model

# Load the model once when the app starts
model = load_intent_model()

@app.route('/nlu/parse', methods=['POST'])
def predict_intent():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "You must provide 'text' in the request"}), 400
    
    text = data['text']
    intent_probs = model.predict_proba([text])[0].tolist()  # Predict probabilities
    max_prob_value = max(intent_probs)
    max_prob_index = intent_probs.index(max_prob_value)
    
    # Define your intent labels
    intent_labels = [
        'information_on_projects',
        'pricing_details',
        'location_details',
        'amenities_and_features',
        'check_availability',
        'schedule_a_visit',
        'reservation_process',
        'option_process',
        'payment_plan'
    ]
    
    intent_name = intent_labels[max_prob_index]
    
    return jsonify({
        "text": text,
        "intent": {
            "name": intent_name,
        }
    })

@app.route('/')
def home():
    return "Intent Detection API is Running"

# if __name__ == '__main__':
#     app.run(debug=True)
