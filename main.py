from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load pre-trained GPT-2 model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate test cases
def generate_test_cases(user_story, num_cases=5, max_length=100):
    input_ids = tokenizer.encode(user_story + "\nTest case examples:", return_tensors="pt")
    test_cases = []

    for _ in range(num_cases):
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        test_case = generated_text.split("Test case examples:")[1].strip()
        test_cases.append(test_case)

    return test_cases

# Sample data for test case classification 
sample_data_file = "sample_data.txt"
sample_data = []
with open(sample_data_file, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        category = parts[1]
        description = parts[0]
        sample_data.append((description, category))

# Prepare data for classification
X, y = zip(*sample_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Function to classify test cases
def classify_test_case(test_case):
    vec = vectorizer.transform([test_case])
    return classifier.predict(vec)[0]

def generate_test_cases_and_save(user_stories, num_cases=5, max_length=100):
    test_cases = []
    for user_story in user_stories:
        generated_test_cases = generate_test_cases(user_story, num_cases, max_length)
        for test_case in generated_test_cases:
            category = classify_test_case(test_case)
            test_cases.append((user_story, test_case, category))

    with open("generated_test_cases.csv", "w") as file:
        for user_story, test_case, category in test_cases:
            test_case = test_case.replace("\n", " ")
            file.write(f"{user_story}|{test_case}|{category}\n")

# Main function to demonstrate the prototype
def main():
    user_stories =  []
    with open("user_stories.txt", "r") as file:
        for line in file:
            user_stories.append(line.strip())
    generate_test_cases_and_save(user_stories)


if __name__ == "__main__":
    main()
