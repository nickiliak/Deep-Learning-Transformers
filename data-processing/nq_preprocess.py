# %% [markdown]
# # Google NQ Dataset

# %% [markdown]
# ## CORPUS DATASET

# %% [markdown]
# reading the original corpus dataset from google nq, filtering by the first 2000 documents and dropping the metadata column ../data_compressed/corpus_filtered.jsonl

# %%
import pandas as pd

# %%
df_nqcorpus = pd.read_json('../data/nq/corpus.jsonl', lines=True)

# %%
df_nqcorpus.head()

# %%
df_nqcorpus["title"].nunique()

# %%
df_nqcorpus_filtered = df_nqcorpus[df_nqcorpus["title"].isin(df_nqcorpus["title"].unique()[:3000])]
#df_nqcorpus_filtered = df_nqcorpuswha

# %%
df_nqcorpus_filtered

# %%
df_nqcorpus_filtered['title'].nunique()

# %%
df_nqcorpus_filtered = df_nqcorpus_filtered.drop(columns=["metadata"])  

# %%
df_nqcorpus_filtered.head()

# %%
df_nqcorpus_filtered.to_json('../data_filtered/corpus_filtered.jsonl', lines=True, orient='records')

# %%
df_nqcorpus_filtered.shape

# %% [markdown]
# ## QUERIES DATASET

# %%
# load
df_nqqueries = pd.read_json('../data/nq/queries.jsonl', lines=True)

# %%
df_nqqueries

# %%
df_nqtest = pd.read_csv('../data/nq/qrels/test.tsv', sep='\t')

# %%
df_nqtest

# %%


# %%
# merge df_nqqueries and df_nqtest by "_id" and "query-id", add to df_nqqueries a new column "corpus-ids" which contains the "corpus-id" values from df_nqtest as a list for each "query-id", and drop metadata and query-id columns
df_nqqueries_merge = df_nqqueries.merge(df_nqtest.groupby('query-id')['corpus-id'].apply(list).reset_index(), left_on='_id', right_on='query-id', how='left')
df_nqqueries_merge = df_nqqueries_merge.drop(columns=['metadata', 'query-id'])

# %%
df_nqqueries_merge

# %%
# now filter df_nqqueries_merge to keep only those rows where all the "corpus-ids" are in df_nqcorpus_filtered
df_nqqueries_merge_filtered = df_nqqueries_merge[df_nqqueries_merge['corpus-id'].apply(lambda ids: all(id_ in df_nqcorpus_filtered['_id'].values for id_ in ids))]

# %%
df_nqqueries_merge_filtered

# %%
# save data to a jsonl in ../data_filtered/queries_filtered.jsonl
df_nqqueries_merge_filtered.to_json('../data_filtered/queries_filtered.jsonl', lines=True, orient='records')


