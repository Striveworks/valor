ALTER TABLE annotation DROP COLUMN embedding_id;

drop index ix_embedding_id;
drop TABLE embedding;
