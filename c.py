import pandas as pd

totalprice = (0)
allproducts = {}


for i in range(2):
    products = input("Enter product name")
    prices = float(input("Enter a product price"))
    allproducts[products]=prices
    totalprice = prices + totalprice
    print(totalprice)


lowestprice = min(allproducts.items(), key=lambda x: x[1])[1]
print("lowestprice", lowestprice)# ('1', 1)


sortedprice = sorted(allproducts.items(), key=lambda x: x[1], reverse=False)

sortedprice.reverse()

fulltotal = totalprice - lowestprice


print("Your Items From Most to Least Expensive:", sortedprice)
print("Cheapest Item:", lowestprice)
print("Your Total Before Discount is:", totalprice)

print("Your Price With Discount is:", fulltotal)
