import chromadb
import phik
import streamlit as st

def column_insights(df):
    insights = []
    for col in df.columns:
        insights.append({
            "column": col,
            "data_type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
            "missing_percentage": round(df[col].isnull().mean() * 100, 2),
            "skewness": round(df[col].skew(), 3) if df[col].dtype in ['int64', 'float64'] else None,
            "duplicates_percentage": round(df[col].duplicated().mean() * 100, 2)
        })
    return insights

def alerts(df):
    alert_list = []
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 20:
            alert_list.append(f"Column '{col}' has {missing_pct:.1f}% missing values.")
        if df[col].dtype in ['int64', 'float64']:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                alert_list.append(f"Column '{col}' is highly skewed (skewness={skewness:.2f}). Consider transformation.")
        duplicates_pct = df[col].duplicated().mean() * 100
        if duplicates_pct > 50:
            alert_list.append(f"Column '{col}' has {duplicates_pct:.1f}% duplicate entries. Consider deduplication.")
    if not alert_list:
        alert_list.append("No critical data quality issues detected.")
    return alert_list

def correlations(df):
    corr_matrix = df.phik_matrix() # using phik for mixed data types
    high_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > 0.7:
                strength = "VERY STRONG" if corr_value > 0.95 else "STRONG"
                action = (
                    "One of these columns should likely be removed (redundant features)."
                    if corr_value > 0.95
                    else "Features are highly related; consider feature selection or PCA."
                )
                high_corrs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(corr_value, 3),
                    "strength": strength,
                    "insight": action
                })
    if not high_corrs:
        return [{"feature_1": "None", "feature_2": "None", "correlation": 0,
                 "strength": "NONE", "insight": "No strong correlations detected"}]
    return high_corrs


def main4(df, l, user_id, cohere_client):
    # Persist client and collection in session state for reuse across interactions
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.Client()
    chroma_client = st.session_state.chroma_client

    collection_name = f"profile_{user_id}"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    # One document per column (better for targeted retrieval and interpretability)
    insights = column_insights(df)
    for i, insight in enumerate(insights):
        col = insight["column"]
        skew_str = f"{insight['skewness']:.3f}" if insight["skewness"] is not None else "N/A"

        # Rich natural language for better semantic embedding
        text = (
            f"Column '{col}' profile: "
            f"data type is {insight['data_type']}. "
            f"Missing values: {insight['missing_percentage']:.1f}% ({insight['null_count']} nulls). "
            f"Unique values: {insight['unique_count']}. "
            f"Skewness: {skew_str}. "
            f"Duplicate entries: {insight['duplicates_percentage']:.1f}%."
        )
        documents.append(text)
        metadatas.append({"category": "column_profile", "column": col})
        ids.append(f"col_{i:03d}_{col[:20]}")  # unique, human readable ID

    # One document per alert (targeted retrieval) 
    alert_messages = alerts(df)
    for i, alert in enumerate(alert_messages):
        documents.append(f"Data quality alert: {alert}")
        metadatas.append({"category": "alert", "column": "all"})
        ids.append(f"alert_{i:03d}")

    # One document per correlation pair 
    corr_data = correlations(df)
    for i, corr in enumerate(corr_data):
        text = (
            f"Correlation insight: '{corr['feature_1']}' and '{corr['feature_2']}' "
            f"have a {corr['strength']} correlation of {corr['correlation']}. "
            f"{corr['insight']}"
        )
        documents.append(text)
        metadatas.append({"category": "correlation", "column": "all"})
        ids.append(f"corr_{i:03d}")

    # Dataset summary (for broad questions) 
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    summary_text = (
        f"Dataset overview: {df.shape[0]} rows and {df.shape[1]} columns. "
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) or 'none'}. "
        f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) or 'none'}. "
        f"Total missing cells: {df.isnull().sum().sum()} "
        f"({df.isnull().mean().mean() * 100:.1f}% of all data)."
    )
    documents.append(summary_text)
    metadatas.append({"category": "dataset_summary", "column": "all"})
    ids.append("dataset_summary_001")

    # --- Embedding all chunks in one batch ---
    embeddings = cohere_client.embed(
        texts=documents,
        model="embed-english-v3.0",
        input_type="search_document"
    ).embeddings

    collection.add(
        embeddings=embeddings,
        documents=documents,   #store the rich natural language insights for interpretability
        metadatas=metadatas,
        ids=ids
    )

    st.session_state.collection = collection
    return collection