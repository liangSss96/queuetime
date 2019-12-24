import pandas as pd
list = ['a', 'b']
temp = pd.DataFrame(list)
temp = temp.apply(lambda x: x+'s')
print(temp)
