features = {}
vals = {}
with open('selected_features_for_all_trials') as f:
    for line in f:
        temp = line.split(", ")
        for ex in temp:
        	ex2 = ex.split(":")
        	ex2[0] = int(ex2[0])
        	ex2[1] = float(ex2[1])
        	if ex2[0] not in features.keys(): 
        		features[ex2[0]] = 1
        		vals[ex2[0]] = ex2[1]
        	else:
    			features[ex2[0]] = features[ex2[0]] + 1
    			if ex2[1] > vals[ex2[0]]:
    				vals[ex2[0]] = ex2[1]

res1 = []
res2 = {}
# print features
for key in features.keys():
	if features[key] >= 8:
		res1.append(key)
		res2[key] = vals[key]
print res1
print res2
