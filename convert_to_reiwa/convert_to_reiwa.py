from sklearn.linear_model import LinearRegression


def hand_made_version(year):
    return year - 2018


xs = ((2019,), (2020,), (2021,), (2022,), (2023,), (2024,), (2025,), (2026,), (2027,))
ys = (1, 2, 3, 4, 5, 6, 7, 8, 9)

model = LinearRegression()
model.fit(xs, ys)


def machine_learning_version(year):
    xs = ((year,),)
    ys = model.predict(xs)

    return ys[0]


for i in range(2019, 2028):
    print(f'西暦{i}年 = 令和{hand_made_version(i)}年')

for i in range(2019, 2028):
    print(f'西暦{i}年 = 令和{machine_learning_version(i)}年')
