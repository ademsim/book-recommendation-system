from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Book Recommender")

BASE = Path(__file__).parent


@st.cache_data
def load_data():
    books = pd.read_csv(BASE / "books_lite.csv")
    neighbors = pd.read_csv(BASE / "book_neighbors.csv")
    return books, neighbors


books, neighbors = load_data()
books = books.sort_values("v", ascending=False).reset_index(drop=True)
labels = (books["title"] + "  —  " + books["authors"].fillna("")).set_axis(books["book_id"])

st.title("Book Recommender")
st.write(
    "Recommendations built from the goodbooks-10k dataset "
    "(about 6 million ratings from 53,000 readers on 10,000 books)."
)

tab_similar, tab_top = st.tabs(["Similar books", "Top books"])

with tab_similar:
    st.write("Pick a book you liked and see which books the same readers liked.")
    book_id = st.selectbox(
        "Book (most popular first, you can type to search)",
        options=books["book_id"],
        format_func=lambda i: labels[i],
    )
    n = st.slider("Number of recommendations", 5, 10, 10, key="n_similar")

    rec = (
        neighbors[neighbors["book_id"] == book_id]
        .sort_values("similarity", ascending=False)
        .head(n)
        .merge(books, left_on="neighbor_id", right_on="book_id", suffixes=("", "_rec"))
    )
    st.dataframe(
        rec[["title", "authors", "similarity", "average_rating"]].rename(columns={"average_rating": "avg rating"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Similarity is the cosine similarity between the rating patterns of two books "
        "(item-based collaborative filtering). Very popular books tend to appear often, "
        "because they share many readers with other books."
    )

with tab_top:
    st.write(
        "Best books for everyone, ranked with a weighted rating: books with few ratings "
        "are pulled toward the overall average, so they cannot dominate the list."
    )
    n_top = st.slider("Number of books", 5, 30, 10, key="n_top")
    one_per_author = st.checkbox("Show only one book per author (avoids series filling the list)", value=True)

    top = books.sort_values("weighted", ascending=False)
    if one_per_author:
        top = top.drop_duplicates("authors")
    st.dataframe(
        top.head(n_top)[["title", "authors", "weighted", "average_rating", "v"]].rename(
            columns={"weighted": "weighted rating", "average_rating": "avg rating", "v": "ratings"}
        ).round(2),
        hide_index=True,
        use_container_width=True,
    )
