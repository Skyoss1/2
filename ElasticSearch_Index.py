from elasticsearch import Elasticsearch, helpers

class ElasticSearchIndex:
    def __init__(self, index_name="yelp_review"):
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        self.index_name = index_name

    def create_index(self, mapping=None):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        self.es.indices.create(index=self.index_name, body=mapping or {})

    def index_documents(self, docs):
        actions = []
        for d in docs:
            doc_id = d.get("review_id")
            if doc_id:
                actions.append({
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": d
                })
        if actions:
            helpers.bulk(self.es, actions)
            self.es.indices.refresh(index=self.index_name)
            print(f"{len(actions)} documents indexed successfully.")
        else:
            print("No valid documents to index.")

    def search(self, query):
        body = {
            "query": {
                "match": {
                    "text": query
                }
            }
        }
        response = self.es.search(index=self.index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            print(f"No documents found for query: {query}")
        return [hit["_source"] for hit in hits]
