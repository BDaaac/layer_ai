from rag import search_law, get_rag_stats

print("=== RAG System Test ===")
print()

# Статистика системы
stats = get_rag_stats()
print(f"📊 Database stats:")
print(f"  Total chunks: {stats.get('total_chunks', 0)}")
print(f"  Total sources: {len(stats.get('sources', []))}")
print(f"  Sources: {', '.join(stats.get('sources', []))}")
print()

# Тестовые поисковые запросы
test_queries = [
    "права потребителя",
    "право собственности", 
    "административная ответственность",
    "Конституция",
    "гражданские права"
]

for query in test_queries:
    print(f"🔍 Query: '{query}'")
    results = search_law(query, k=2)
    
    if results:
        for i, result in enumerate(results, 1):
            source = result['metadata']['source']
            content = result['chunk'][:150] + "..."
            score = result['score']
            print(f"  {i}. {source} (score: {score:.3f})")
            print(f"     {content}")
    else:
        print("  No results found")
    print()