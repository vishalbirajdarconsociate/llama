from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()
while True:
    inp = input("search : ")
    print(search.invoke(inp))