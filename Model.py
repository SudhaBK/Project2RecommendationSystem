import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

class HybridRecommender:
    def __init__(self, df, num_dimensions=10):
        self.df = df
        self.num_dimensions = num_dimensions
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
        self.df['rating'] = self.df['rating'].fillna(0)
        self.user_product_matrix = self.create_user_product_matrix()
        self.svd_matrix = self.create_svd_matrix()
        self.nbrs = self.create_nbrs()

    def create_user_product_matrix(self):
        return self.df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

    def create_svd_matrix(self):
        svd = TruncatedSVD(n_components=self.num_dimensions)
        return svd.fit_transform(self.user_product_matrix)

    def create_nbrs(self):
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
        nbrs.fit(self.svd_matrix)
        return nbrs

    def predict(self, user_id, num_recommendations):
        idx = self.df[self.df['user_id'] == user_id].index[0]
        user_vector = self.svd_matrix[idx].reshape(1, -1)
        distances, indices = self.nbrs.kneighbors(user_vector, n_neighbors=num_recommendations+1)
        product_indices = indices[0][1:]
        recommended_products = self.df.iloc[product_indices][['product_name', 'about_product', 'img_link', 'discounted_price', 'rating', 'discount_percentage']].drop_duplicates(subset='product_name', keep='first')
        return recommended_products.head(num_recommendations)