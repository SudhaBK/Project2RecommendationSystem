# Importing the necessary libraries
import streamlit as st
import pandas as pd
from Model import HybridRecommender
import plotly.express as px

# Creating a HybridRecommender object with the Amazon dataset
recommender = HybridRecommender(pd.read_csv('amazon.csv'))

# Here we get the top 20 unique user names from the dataset
top_users = [name.strip().split(',')[0].strip() for name in recommender.df['user_name'].unique()[:20]]

# Custom CSS function to add background image and styling purpose
def add_background():
    # Add custom CSS to the our App
    st.markdown(
        f"""
        <style>
        /* Background Image */
        .stApp {{
            background: url("https://www.shutterstock.com/image-photo/one-star-clicks-businessman-concept-260nw-769083079.jpg") no-repeat center center fixed; 
            background-size: cover;
        }}
        /* Text Styling */
        h1, h2, h3, h4, h5, h6 {{
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }}
        p, label, .stMarkdown {{
            color: white;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        }}
        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.7);
        }}
        .css-1p1n3ar {{
            color: white;
        }}
        /* Heading Color */
        .title {{
            color: #786C3B;
        }}
        /* Label Font Size */
        label {{
            font-size: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Adding background CSS to our  app
add_background()

# Creating a selectbox to select the User name(Top 20 which are in database)
st.markdown("<h1 class='title'>Product Recommendation Hub</h1>", unsafe_allow_html=True)
user_name = st.selectbox("Select User Name", top_users, key=None)

# We get here the number of recommendations from the user
num_recommendations = st.slider("Number of Recommendations", 1, 10)

# Just to better look, changed Font color and background color of the "Recommend" button
st.markdown(
    """
    <style>
    .stButton > button:first-child {
        color: #ffffff;
        background-color: #000000;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Making the predictions and displaying the results when the "Recommend" button is clicked
if st.button('Recommend'):
    # Get the user ID from the selected user name
    user_id_df = recommender.df[recommender.df['user_name'].str.strip().str.lower().str.split(',').str[0].str.strip() == user_name.strip().lower()]
    
    if not user_id_df.empty:
        user_id = user_id_df['user_id'].values[0]
        
        #Using HybridRecommender we are making the predictions
        recommended_products = recommender.predict(user_id, num_recommendations)
        
        if recommended_products is not None and not recommended_products.empty:
            # Preprocessing the recommended products data here
            recommended_products['discounted_price'] = recommended_products['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
            recommended_products['discount_percentage'] = recommended_products['discount_percentage'].str.replace('%', '').astype(float)
            recommended_products['actual_price'] = recommended_products.apply(lambda row: row['discounted_price'] / (1 - row['discount_percentage'] / 100), axis=1)

            # Here we are fdisplaying the recommended products
            st.write("Recommended Products:")
            for index, row in recommended_products.iterrows():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(row['img_link'], width=200, caption=row['product_name'])
                with col2:
                    # Preprocess the product name, as product name is longer string
                    product_name = row['product_name']
                    words_to_remove = ['Compatible for', 'for iPhone', 'for iPad', 'for Android']
                    for word in words_to_remove:
                        product_name = product_name.replace(word, '')
                    product_name = product_name.strip()
                    
                    # Display the product details
                    st.markdown(f"**<font size='4' color='white'>{product_name}</font>**", unsafe_allow_html=True)
                    st.markdown(f"**<font size='4' color='orange'>Price:</font>** <font size='4'>{row['discounted_price']}</font>", unsafe_allow_html=True)
                    st.markdown(f"**<font size='4' color='orange'>Rating:</font>** <font size='4'>{row['rating']}</font></font>", unsafe_allow_html=True)
                    st.markdown(f"**<font size='4' color='orange'>Discount:</font>** <font size='4'>{row['discount_percentage']}%</font></font>", unsafe_allow_html=True)
                    
            # Create a px-bar chart to display the product ratings
            # px.bar() allows you to create an interactive, web-based bar chart that is easy to customize
            fig = px.bar(recommended_products, x='product_name', y='rating', 
                           hover_data=['discounted_price', 'actual_price', 'discount_percentage'], 
                           color='rating')
            
            # Update the hover template to display the product details(we want to display additional information about that data point)
            hover_text = recommended_products.apply(lambda row: f"Discounted Price: ₹{row['discounted_price']:.2f}<br>Actual Price: ₹{row['actual_price']:.2f}<br>Discount: {row['discount_percentage']}%", axis=1)
            fig.update_traces(hovertext=hover_text, hovertemplate="Product Name: %{x}<br>Rating: %{y}<br>%{hovertext}")
            
           # Customize the chart's visual appearance and Set chart title, axis labels, and other layout options
            fig.update_layout(title='Product Ratings', xaxis_title='Product Name', yaxis_title='Rating')
            
            # Show the chart or the Plot display
            st.plotly_chart(fig)