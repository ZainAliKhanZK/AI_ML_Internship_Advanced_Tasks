import torch
import app as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "ZainAliKhanZAK/bert_agnews_classifier"          # fine-tuned model weights live here
TOKENIZER_SOURCE = "ZainAliKhanZAK/bert_agnews_classifier"    
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

LABEL_EMOJI = {
    "World": "🌍",
    "Sports": "🏅",
    "Business": "💼",
    "Sci/Tech": "🔬",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)  
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model


def predict(text, tokenizer, model):
    device = model.device
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    return probs


st.set_page_config(page_title="News Topic Classifier", page_icon="📰")
st.title("📰 News Topic Classifier (BERT)")
st.write(
    "Fine-tuned `bert-base-uncased` on the AG News dataset (94.6% test accuracy). "
    "Paste a news headline or short snippet below to classify it."
)

tokenizer, model = load_model()

text_input = st.text_area(
    "News headline / snippet",
    placeholder="e.g. Apple unveils new AI chip for next-gen iPhones",
    height=100,
)

if st.button("Classify", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Classifying..."):
            probs = predict(text_input, tokenizer, model)

        results = sorted(
            [(LABEL_NAMES[i], probs[i]) for i in range(4)],
            key=lambda x: x[1],
            reverse=True,
        )

        top_label, top_score = results[0]
        emoji = LABEL_EMOJI.get(top_label, "")
        st.success(f"**Predicted category: {emoji} {top_label}** ({top_score*100:.1f}% confidence)")

        st.subheader("Confidence breakdown")
        for label, score in results:
            emoji = LABEL_EMOJI.get(label, "")
            st.write(f"{emoji} {label}")
            st.progress(float(score))

st.caption(
    "Note: this model performs best on longer, news-style text (similar to Reuters-style headlines "
    "+ descriptions), matching its AG News training data."
)

# Some example headlines for testing:

# World
# "United Nations Security Council to vote on new sanctions against rogue state"
# "Earthquake kills dozens in remote mountain region, rescue efforts underway"
# "Prime Minister announces surprise cabinet reshuffle amid coalition tensions"

# Sports
# 4. "NEW YORK (Reuters) - The New York Yankees clinched a playoff spot on Sunday after a dramatic ninth-inning comeback against the Boston Red Sox."
# 5. "Olympic committee confirms host city for 2036 Summer Games"
# 6. "Star striker ruled out for the season with knee injury ahead of championship match"

# Business
# 7. "LONDON (Reuters) - Shares in the retail sector tumbled on Monday after disappointing quarterly earnings from several major chains."
# 8. "Central bank raises interest rates for the third consecutive quarter to curb inflation"
# 9. "Tech giant announces layoffs affecting thousands of employees worldwide"

# Sci/Tech
# 10. "Researchers unveil new battery technology that could double electric vehicle range"
# 11. "SAN FRANCISCO (Reuters) - A group of scientists announced a breakthrough in quantum computing that could accelerate drug discovery."
# 12. "Space agency delays satellite launch due to technical malfunction"

# Ambiguous / tricky (good for testing edge cases)
# 13. "Sports league signs record-breaking broadcast rights deal worth billions" (Sports vs Business overlap)
# 14. "Government unveils new AI regulation policy following industry pressure" (World vs Sci/Tech overlap)
# 15. "Cricket board announces new international tournament format" (World vs Sports overlap)