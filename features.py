import Executesql as es
data = es.origindata()
print(data.head(10))

for kind,subkind in data.groupby(data['KIND_NAME']):
    print(kind,len(subkind))