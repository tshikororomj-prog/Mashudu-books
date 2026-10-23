import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_url = "https://books.toscrape.com/catalogue/page-{}.html"
headers = {"User-Agent": "Mozilla/5.0"}
books = []
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

for page in range(1, 51):
    url = base_url.format(page)
    for article in soup.find_all('article', class_='product_pod'):
        title = article.h3.a['title']
        price_text = article.find('p', class_='price_color').text
        rating_word = article.find('p', class_='star-rating')['class'][1]
        rating = rating_map.get(rating_word, 0)
        availability = article.find('p', class_='instock availability').text.strip()
        detail_url = "https://books.toscrape.com/catalogue/" + article.h3.a['href']

        books.append([title, price, rating_word, rating, availability, quantity, category])

df = pd.DataFrame(books, columns=['title', 'price_gbp', 'rating_word', 'rating_num', 'availability_text', 'quantity',
                                  'category'])
print(f"Collected {len(df)} books")
df.head()

df.to_csv("books_raw.csv", index=False)

print(df.isnull().sum())

df['title'] = df['title'].str.strip()

df['in_stock'] = df['availability_text'].apply(lambda x: 1 if 'In stock' in x else 0)
df['title_length'] = df['title'].apply(len)
df['word_count'] = df['title'].apply(lambda x: len(x.split()))
print(df.describe())

df_clean = df[
    ['title', 'price_gbp', 'rating_num', 'quantity', 'category', 'in_stock', 'title_length', 'word_count']].copy()
df_clean.to_csv("books_cleaned.csv", index=False)
df_clean.head()

plt.figure(figsize=(8, 5))
plt.title("Distribution of Book Prices (£)")
plt.xlabel("Price");
plt.savefig("price_dist.png");
plt.show()

plt.figure(figsize=(8, 5))
plt.title("Price vs Star Rating");
plt.savefig("price_vs_rating.png");
plt.show()

plt.figure(figsize=(12, 6))
order = df_clean.groupby('category')['price_gbp'].mean().sort_values(ascending=False).index
plt.title("Average Price by Category");
plt.tight_layout();
plt.savefig("price_by_category.png");
plt.show()

plt.figure(figsize=(6, 4))
plt.title("Title Length vs Price coloured by Rating");
plt.savefig("title_vs_price.png");
plt.show()

X = df_clean[['rating_num', 'quantity', 'title_length', 'word_count', 'category']]
y = df_clean['price_gbp']

X = pd.get_dummies(X, columns=['category'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# Coefficients
coef_df = pd.DataFrame({'feature': X.columns, 'coef': model.coef_}).sort_values('coef', ascending=False)
print(coef_df.head(10))

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price");
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price");
plt.savefig("actual_vs_pred.png");
plt.show()

results = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
results.to_csv("predictions.csv", index=False)