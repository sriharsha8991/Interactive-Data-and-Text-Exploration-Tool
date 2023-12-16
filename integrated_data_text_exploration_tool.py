import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

def main():
    st.title('Advanced Interactive Data and Text Exploration Tool')

    # File uploader for CSV and Excel
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        df = read_file(uploaded_file)
        st.write("Data Preview:", df.head())
        filtered_data = generate_filters(df)

        if st.checkbox('Perform Text Analysis'):
            textual_analysis(df)

        visualize_data(filtered_data)

def read_file(uploaded_file):
    if uploaded_file.type == "text/csv":
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def generate_filters(df):
    st.sidebar.title("Filters")
    filtered_df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = df[col].min().item(), df[col].max().item()
            values = st.sidebar.slider(f"Filter by {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
            filtered_df = filtered_df[filtered_df[col].between(values[0], values[1])]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            min_date, max_date = df[col].min(), df[col].max()
            date_range = st.sidebar.date_input(f"Filter by {col}", [min_date, max_date])
            filtered_df = filtered_df[filtered_df[col].between(*date_range)]
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            options = st.sidebar.multiselect(f"Filter by {col}", df[col].unique(), default=df[col].unique())
            filtered_df = filtered_df[filtered_df[col].isin(options)]

    st.write("Filtered Data:", filtered_df)
    return filtered_df

def textual_analysis(df):
    st.subheader("Text Analysis")
    text_column = st.selectbox("Select Text Column for Analysis", df.select_dtypes(include='object').columns)
    if text_column:
        generate_wordcloud(df[text_column])
        perform_sentiment_analysis(df, text_column)

def generate_wordcloud(text_series):
    text = ' '.join(text_series.dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def perform_sentiment_analysis(df, column):
    st.subheader("Sentiment Analysis")
    df['Sentiment'] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    st.bar_chart(df['Sentiment'])

def visualize_data(df):
    st.title("Customizable Data Visualizations")
    if not df.empty:
        viz_type = st.selectbox("Select Visualization Type", ['Histogram', 'Count Plot', 'Scatter Plot', 'Heatmap', 'Box Plot'])
        
        if viz_type in ['Histogram', 'Scatter Plot', 'Box Plot']:
            numeric_columns = df.select_dtypes(['float', 'int']).columns.tolist()
            column = st.selectbox("Select Numeric Column", numeric_columns)
            
            if column and column in df.columns:
                if viz_type == 'Histogram':
                    plot_histogram(df, column)
                elif viz_type == 'Scatter Plot':
                    second_column = st.selectbox("Select Second Numeric Column", numeric_columns)
                    plot_scatter_plot(df, column, second_column)
                elif viz_type == 'Box Plot':
                    plot_box_plot(df, column)
        elif viz_type == 'Count Plot':
                    categorical_columns = df.select_dtypes(['object', 'category']).columns.tolist()
                    column = st.selectbox("Select Categorical Column", categorical_columns)
                    plot_count_plot(df, column)
        elif viz_type == 'Heatmap':
                    plot_heatmap(df)

def plot_histogram(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)

def plot_scatter_plot(df, column1, column2):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=column1, y=column2, ax=ax)
    st.pyplot(fig)

def plot_box_plot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=column, ax=ax)
    st.pyplot(fig)

def plot_count_plot(df, column):
    if column in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=df[column], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.error("Selected column not found in the DataFrame.")

def plot_heatmap(df):
    # st.write("DataFrame shape:", df.shape)  # Debugging information

    # # Simplify the plot for debugging
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # st.write("Numeric columns:", numeric_cols)  # Debugging information

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
    #     st.write("Correlation matrix:", corr_matrix)  # Debugging information

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True)
        plt.show()
        st.pyplot()
    else:
        st.error("Not enough numerical data for a heatmap.")


if __name__ == "__main__":
    main()
