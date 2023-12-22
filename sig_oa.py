def hashMap(queryType, queries):
    keypointer = 0
    valpointer = 0
    mp = {}
    ans = 0
    
    for i in range(len(queryType)):
        if queryType[i] == "insert":
            mp[queries[i][0] - keypointer] = queries[i][1] - valpointer
        elif queryType[i] == "addToValue":
            valpointer += queries[i][0]
        elif queryType[i] == "addToKey":
            keypointer += queries[i][0]
        else:
            ans += mp.get(queries[i][0] - keypointer, -1 * valpointer) + valpointer
    
    return ans

    
queryType = ["insert","addToValue","get","insert","addToKey","addToValue","get"]
queries = [[1,2],[2],[1],[2,3],[1],[-1],[3]]

res = hashMap(queryType,queries)
print(res)